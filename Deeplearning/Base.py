import  os
import  torch
import  time
import  torch.nn    as      nn
import  pandas      as      pd
from    torch.optim import  lr_scheduler
from    tqdm        import  tqdm

    
class AverageMeter(object):
    """
    computes and stores the average and current value
    """
    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def normal_accuracy(pred,labels):    
    return ((pred.argmax(dim=1)==labels).sum()/len(labels))*100

def teacher_forcing_decay(epoch, num_epochs):
    initial_tf_ratio = 1.0
    final_tf_ratio = 0.01
    decay_rate = (final_tf_ratio / initial_tf_ratio) ** (1 / (num_epochs - 1))

    tf_ratio = max(0.01,initial_tf_ratio * (decay_rate ** epoch))
    return tf_ratio



def train(
    
    data_tensor                 : torch.tensor,
    prediction_input_size       : int,
    prediction_horizon         : int,
    _divition_factr             :int,

    model                       :torch.nn.Module,
    model_name                  :str,
    epochs                      :int,
    load_saved_model            :bool,
    ckpt_save_freq              :int ,
    ckpt_save_path              :str ,
    ckpt_path                   :str ,
    report_path                 :str ,
    
    criterion ,
    optimizer,
    lr_scheduler,
    sleep_time,
    Validation_save_threshold : float ,
    device                      :str        = 'cuda'    ,
    if_validation = False
    ):

    model       = model.to(device)

    if load_saved_model:
        model, optimizer = load_model(ckpt_path=ckpt_path, model=model, optimizer=optimizer)

    report = pd.DataFrame(
                            columns=[
                                    "model_name",
                                    "mode",
                                    "image_type",
                                    "epoch",
                                    "learning_rate",
                                    "batch_size",
                                    "batch_index",
                                    "loss_batch",
                                    "avg_train_loss_till_current_batch",
                                    "avg_val_loss_till_current_batch",
                                    ])
    
    #############################################################3#params
    ll = (data_tensor.shape[-1]//prediction_input_size)-1
    acc1 = 0
    for epoch in tqdm(range(1, epochs + 1)):
        loss_avg_train  = AverageMeter()
        loss_avg_val    = AverageMeter()
        model.train()
        mode = "train"
        
        loop_train = tqdm(  range(ll),
                            total=ll,
                            desc="train",
                            position=0,
                            leave=True
                        )
        accuracy_dum=[]
        for batch_idx in loop_train:
            
            optimizer.zero_grad()

            x = data_tensor[batch_idx*prediction_input_size    :(batch_idx+1)*prediction_input_size]+((2*torch.rand(size=[prediction_input_size],device=device)-1)/(_divition_factr*2))
            y = data_tensor[(batch_idx+1)*prediction_input_size:(batch_idx+1)*prediction_input_size+prediction_horizon]


            prediction_list = torch.zeros(size=[prediction_horizon]).to(device)
            decoder_hidden, decoder_cell = torch.zeros(size=[2,prediction_input_size],device=device), torch.zeros(size=[2,prediction_input_size],device=device)
            for i in range(prediction_horizon):
                # prediction = inception.forward(x)
                prediction,(decoder_hidden, decoder_cell) = model.forward(x.unsqueeze(0),decoder_hidden, decoder_cell)#
                x =  torch.cat([x[1:],prediction],dim=0)
                prediction_list[i] = prediction
            loss = criterion(prediction_list, y,model.Koopman_operator.weight)

            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
        
                     
            # gradient clipping
            # max_grad_norm = 1.0
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    

            length = x.shape[0]
            loss_avg_train.update(loss.item(), length)
            
            new_row = pd.DataFrame(
                                    {
                                    "model_name": model_name,
                                    "mode": mode,
                                    "image_type":"original",
                                    "epoch": epoch,
                                    "learning_rate":optimizer.param_groups[0]["lr"],
                                    "batch_size":length,
                                    "batch_index": batch_idx,
                                    "loss_batch": loss.detach().item(),
                                    "avg_train_loss_till_current_batch":loss_avg_train.avg,
                                    "avg_val_loss_till_current_batch":None,
                                    },
                                    index=[0])

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                # accuracy_train="{:.4f}".format(acc1),
                refresh=True,
            )
        # time.sleep(3)
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )
        # if utils.scheduler_activate:    
        #     lr_schedulerr.step()
        if if_validation:    
            model.eval()
            mode = "val"
            with torch.no_grad():
                loop_val = tqdm(
                    range(ll),
                    total=ll,
                    desc="val",
                    position=0,
                    leave=True,
                )
                acc1 = 0

                for batch_idx in loop_val:
                    optimizer.zero_grad()
                                    
                    x = data_tensor[batch_idx*prediction_input_size    :(batch_idx+1)*prediction_input_size]+(((2*torch.rand(size=[prediction_input_size],device=device)-1)/(_divition_factr*2)))
                    y = data_tensor[(batch_idx+1)*prediction_input_size:(batch_idx+1)*prediction_input_size+prediction_horizon]

                    prediction_list = torch.zeros(size=[prediction_horizon]).to(device)
                    decoder_hidden, decoder_cell = torch.zeros(size=[2,prediction_input_size],device=device), torch.zeros(size=[2,prediction_input_size],device=device)
                    for i in range(prediction_horizon):
                        # prediction = inception.forward(x)
                        prediction,(decoder_hidden, decoder_cell) = model.forward(x.unsqueeze(0),decoder_hidden, decoder_cell)#
                        x =  torch.cat([x[1:],prediction],dim=0)
                        prediction_list[i] = prediction
                    loss = criterion(prediction_list, y,model.Koopman_operator.weight)

                    acc1 = 0
                    length = x.shape[0]
                    loss_avg_train.update(loss.item(), length)
                
                
                    new_row = pd.DataFrame(
                                            {
                                            "model_name": model_name,
                                            "mode": mode,
                                            "image_type":"original",
                                            "epoch": epoch,
                                            "learning_rate":optimizer.param_groups[0]["lr"],
                                            "batch_size": length,
                                            "batch_index": batch_idx,
                                            "loss_batch": loss.detach().item(),
                                            "avg_train_loss_till_current_batch":None,
                                            "avg_val_loss_till_current_batch":loss_avg_val.avg,
                                            },
                                            index=[0],)
                    
                    report.loc[len(report)] = new_row.values[0]
                    loop_val.set_description(f"val - iteration : {epoch}")
                    loop_val.set_postfix(
                        loss_batch="{:.4f}".format(loss.detach().item()),
                        avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                        # accuracy_val="{:.4f}".format(acc1),
                        refresh=True,
                    )
            
    report.to_csv(os.path.join(report_path,f"{model_name}_report.csv"))
    torch.save(model.state_dict(), os.path.join(report_path,f"{model_name}.pt"))
    return model, optimizer, report