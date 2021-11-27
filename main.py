from neural_net import PV_Net


cnn = PV_Net()
cnn.init_datasets("/home/lg08/main/data/standard_collective", 0.9, "extras/solar_array")
cnn.train()
cnn.save_net()
cnn.load_net()
cnn.test()
cnn.test_whole()
