# scales = np.arange(1, Utils.Scales)
# import pywt
# for wavelet in reversed(Utils.wavelet_lis):
#     # Perform Continuous Wavelet Transform
#     print(wavelet)
#     coefficients, frequencies = pywt.cwt(Dataloader.X_numpy[1,:Utils.input_horizon], scales, wavelet)

#     # # Plot the results
#     # plt.figure(figsize=(10, 6))
#     # plt.imshow(np.abs(coefficients), extent=[0, 1, 1, Utils.Scales], aspect='auto', cmap='jet')
#     # plt.colorbar(label='Magnitude')
#     # plt.title('Continuous Wavelet Transform')
#     # plt.xlabel('Time')
#     # plt.ylabel('Scale')
#     # plt.show()

# if Utils.Coefficient_Real:
#     coefficients = np.abs(coefficients)