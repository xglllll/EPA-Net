# EPA-Net: An Efficient and Lightweight Pyramid Attention for Image Deblurring


#### News

- **OCT 11, 2025:** Paper accepted at Pattern Recognition :tada:


<hr />

> **Abstract:** *The effective use of multiscale information can improve image deblurring performance. Although several multiscale attentions have been proposed, the large overhead introduced by these attentions limits their widespread application in image deblurring networks. In this paper, we present an efficient and lightweight pyramid attention (EPA), which introduces slight additional parameters and computational costs. Briefly, the EPA first employs the lightweight channel Attention (LCA) module to capture global information. Then, lightweight depthwise convolution is used to extract multiscale features. Finally, we use a simplified channel self-attention, name as DTFF module, to aggregate information at different scales. Unlike traditional channel self-attention, DTFF module adopts two unique techniques, replacing Q (Query) or K (Key) with original input, and using channel anchored attention to dynamically aggregate information at different scales. To further demonstrate the potential of EPA, we also designed an efficient EPA-based network (EPA-Net). The experimental results show that EPA produces consistent improvements over the existing deblurring networks with small additional overheads, and the EPA-Net achieves the best tradeoff between effectiveness and efficiency. The source code is available on: https://github.com/xglllll/EPA-Net* 
<hr />

## Network Architecture

<img src = "https://i.imgur.com/A1ycZgV.jpeg"> 


## Datasets
<table>
<thead>
  <tr>
    <th>GoPro</th>
    <th>HIDE</th>
    <th>RealBlur_J</th>
    <th>RealBlur_R</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align:center;"> <a href="https://drive.google.com/drive/folders/1AsgIP9_X0bg0olu2-1N6karm2x15cJWE?usp=sharing">train</a> </td>
    <td style="text-align:center;"> - </td>
    <td style="text-align:center;"> - </td>
    <td style="text-align:center;"> - </td>
  </tr>

  <tr>
    <td style="text-align:center;"> <a href="https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf?usp=sharing">test</a> </td>
    <td style="text-align:center;"> <a href="https://drive.google.com/drive/folders/1nRsTXj4iTUkTvBhTcGg8cySK8nd3vlhK?usp=sharing">test</a> </td>
    <td style="text-align:center;"> <a href="https://drive.google.com/drive/folders/1KYtzeKCiDRX9DSvC-upHrCqvC4sPAiJ1?usp=sharing">test</a> </td>
    <td style="text-align:center;"> <a href="https://drive.google.com/drive/folders/1EwDoajf5nStPIAcU4s9rdc8SPzfm3tW1?usp=sharing">test</a> </td>
  </tr>
</tbody>
</table>


## Training
- Train the model with default arguments by running

```
python train.py
```

## Evaluation

### Download the [model](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view?usp=sharing) and place it in ./pretrained_models/

#### Testing on GoPro dataset
- Download [images](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf?usp=sharing) of GoPro and place them in `./Datasets/GoPro/test/`
- Run
```
python test.py --dataset GoPro
```

#### Testing on HIDE dataset
- Download [images](https://drive.google.com/drive/folders/1nRsTXj4iTUkTvBhTcGg8cySK8nd3vlhK?usp=sharing) of HIDE and place them in `./Datasets/HIDE/test/`
- Run
```
python test.py --dataset HIDE
```


#### Testing on RealBlur-J dataset
- Download [images](https://drive.google.com/drive/folders/1KYtzeKCiDRX9DSvC-upHrCqvC4sPAiJ1?usp=sharing) of RealBlur-J and place them in `./Datasets/RealBlur_J/test/`
- Run
```
python test.py --dataset RealBlur_J
```



#### Testing on RealBlur-R dataset
- Download [images](https://drive.google.com/drive/folders/1EwDoajf5nStPIAcU4s9rdc8SPzfm3tW1?usp=sharing) of RealBlur-R and place them in `./Datasets/RealBlur_R/test/`
- Run
```
python test.py --dataset RealBlur_R
```

#### To reproduce PSNR/SSIM scores of the paper on GoPro and HIDE datasets, run this MATLAB script
```
evaluate_GOPRO_HIDE.m 
```

#### To reproduce PSNR/SSIM scores of the paper on RealBlur dataset, run
```
evaluate_RealBlur.py 
```


## Performance

<img src= "https://i.imgur.com/i51aJxj.png">


|   Dataset   | GoPro     | HIDE       | RealBlur-J |
| :---------: | :-------: | :--------: | :---------: |
|  PSNR (dB)  |   33.22   |   31.10    |    28.82    |
|    SSIM     |   0.963   |   0.941    |    0.879    |



## Citation
If you use EPA-Net, please consider citing:

    @article{hua2025efficient,
        title={An efficient and lightweight pyramid attention for image deblurring},
        author={Hua, Xia and Xiang, Guoliang and Yuan, Haiwen and Zou, Lu and Wang, Lei and Hong, Hanyu},
        journal={Pattern Recognition},
        pages={112506},
        year={2025},
        publisher={Elsevier}
    }


## Acknowledgment
This code borrows heavily from [Uformer](https://github.com/ZhendongWang6/Uformer). Thanks for their awesome work.


## Contact
Please contact us if there is any question or suggestion(Xia Hua hedahuaxia05021046@163.com, Guoliang Xiang xgl9450430@gmail.com).