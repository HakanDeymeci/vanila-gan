# vanila-gan
Teamproject GAN Group 1.
Team members: Julien, Hakan, Dennis, Delight

Documentation:<br />
#21.05.2020<br />
We created a Whatsapp Group and a Discord sever to be better in contact with each of us.<br />

#25.05.2020<br />
Over the weekend, we created a text-channel on our Discord server "websites-and-links" where we collected some useful links and explanations about our project.<br />
The Monday later we had our first meeting over Discord. We created the repository and the project "vanila-gan" and tried to get used to github.<br />
Results from the first Meeting:<br />
	- created the project<br />
	- Implemented during the meeting the "random_noise(dim,batch_size)" and the "data_loader" function<br />
	- Distribution of tasks:<br />
		- Discriminator research & implementation Hakan and Julien<br />
		- Generator research & implementation Delight and Dennis<br />
		- created a new experimental branch<br />

#26.05.2020 - #30.05.2020 <br />
---research time---<br />

#31.05.2020<br />
Later this week we had our second meeting, where we presented and explained our tasks to the other duo.<br />
We put an sturcture for the Generator and the Discriminator in the experimental branch.<br />
Results from the second Meeting:<br />
	- updated and adjusted the discriminator/generator layout <br />
	- added the train function<br />

#03.06.2020<br />
Results from the third Meeting:<br />
	- added the optim.Adam() optimizers in the experimental branch<br />

#10.06.2020<br />
We faced during the Meeting some size-mismatch problems which we tried to remedy for the rest of the day.<br />
Results from the fourth Meeting:<br />
	- updated the "data_loader" function<br />
	- added the converter functions img_to_vec and vec_to_img in the experimental branch<br />
	- added code for importing the logger from 'Google Drive'<br />
	- got our first visual outputs<br />

#11.06.2020 - now<br />
After we had some good results without any errors in our experimental branch we started impelmenting parts of the code in our master Branch.<br />
We added some variables for functions and numbers for better visualization.<br />




Tutorial on how to run the GAN:<br />

1. Go on Colab using the following link 'https://colab.research.google.com/notebooks/welcome.ipynb'.<br />
2. Press on the 'File/Open Notebook'. You can also use the hotkeys 'STRG+O'.<br />
3. Press GitHub on the upper bar and wait for colab to authorize with your GitHub account.<br />
4. Now you choose 'HakanDeymeci/vanila-gan' as Repository and 'master' as Branch.<br />
5. Now open the python notebook data called 'Generative_Adversarial_Networks_PyTorch.ipynb'.<br />
6. Importing the Logger file from Drive:<br />
6.1. Go on the following link 'https://github.com/HakanDeymeci/vanila-gan/blob/master/utils.py' and<br />
download this file.<br />
6.2. Add this file to Drive over the link 'https://drive.google.com/drive/my-drive'.<br />
6.3. Press hotkeys 'STRG+F9' and run the code for the first time.<br />
6.4. Press on the link below the first code snippet to get authorization code.<br />
6.5. Copy this code and insert it in the field and press enter.<br />
6.6. Now tensorboardX-2.0 should be installed successfully.<br />
7. To run the GAN press now the hotkeys 'STRG+F9' oder go under 'Run all' over the dropdown menu
'Runtime'.<br />





	

 




