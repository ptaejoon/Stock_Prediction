import pickle
import stockGAN
import os


ganFileName = "stockGAN.sav"
if os.path.isfile(ganFileName):
    print("Saved GAN Model Already Exists. Loading stockGAN.sav")
    file = open(ganFileName, 'rb')
    gan = pickle.load(file)
else:
    print("Saved GAN Model Doesn't Exists. Saving stockGAN.sav")
    file = open(ganFileName,'wb')
    sav_gan = stockGAN.GAN(batch_size=20)
    sav_gan.train()
    pickle.dump(sav_gan,file)
    gan = sav_gan

predicted_2019 = gan.predict_testSet()
answer_2019 = gan.GAN_testY
print(predicted_2019)
print(gan.generator.evaluate(gan.GAN_testX,gan.GAN_testY))




