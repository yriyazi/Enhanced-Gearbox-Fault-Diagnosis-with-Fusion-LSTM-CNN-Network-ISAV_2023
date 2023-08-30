
loss_t = []

ll = (data_tensor.shape[-1]//prediction_input_size)-1
for epoch in tqdm.tqdm(range(1,epochs)):
    loop_train = tqdm.tqdm(range(ll),total=ll,desc="train",position=0,leave=True)
    
    loss_train_list = []
    
    for Batch in loop_train:
        optimizer.zero_grad()
        
        x = data_tensor[Batch*prediction_input_size:(Batch+1)*prediction_input_size]+(torch.rand(size=[prediction_input_size],device=device)/_divition_factr)
        _x = x
        y = data_tensor[(Batch+1)*prediction_input_size:(Batch+1)*prediction_input_size+prediction_horizion]

        prediction_list = torch.zeros(size=[prediction_horizion]).to(device)

        decoder_hidden, decoder_cell = torch.zeros(size=[2,prediction_input_size],device=device), torch.zeros(size=[2,prediction_input_size],device=device)
        for i in range(prediction_horizion):
            # prediction = inception.forward(x)
            prediction,(decoder_hidden, decoder_cell) = inception.forward(x.unsqueeze(0),decoder_hidden, decoder_cell)#
            x =  torch.cat([x[1:],prediction],dim=0)
            prediction_list[i] = prediction


        loss_train = criterion(prediction_list, y)
        # Back propagation
        loss_train.backward()
        # Update model parameters
        optimizer.step()
        loss_train_list.append(loss_train)
        
        if Batch%10 == 0:
            loop_train.set_description(f"Train {epoch}- iteration : {Batch}")
            loop_train.set_postfix(
                loss_batch="{:.7f}".format(torch.tensor(loss_train_list).mean()),refresh=True,)
            
    loss_t.append(torch.tensor(loss_train_list).mean())