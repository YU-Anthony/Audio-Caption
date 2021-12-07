# Audio-Caption
This is a demo of DCASE 2020 Challenge T6 task which used the pre-trained model came from Project [dcase_2020_T6](https://github.com/lukewys/dcase_2020_T6).

## 1. Set-up code
To set-up the code, you have to do the following:

1. Clone this repository.
2. Use either pip or conda to install dependencies

Firstly, use the following command to clone this repository at your terminal:
```bash
$ git clone git@github.com:audio-captioning/dacse-2020-baseline.git
```
The, create a new conda environment

```bash
$ conda create -n dcase_T6 python
```
The above command will create a new environment called `dcase_T6`.
```bash
$ conda activate dcase_T6
```
After you change the conda enviroment to the one you created just now, it's time to install dependencies:
```bash
pip install -r requirements.txt
```
## 2. Run the demo
You just need to run the `demo.ipynb` to run the project. This file used `Gradio` framework to build a easy demo to display the power of your model.
![image](https://user-images.githubusercontent.com/57721340/144982917-fe7ec62d-f8a2-439a-b468-d77ebd682495.png)

