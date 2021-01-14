# LAEO-Net++
_Code coming soon_  

See previous version [here](https://github.com/AVAuco/laeonet)

Support code for [LAEO-Net++ paper](https://arxiv.org/abs/2101.02136) (IEEE TPAMI, 2021).

<div align="center">
    <img src="./LAEOnetplus.png" alt="The LAEO-Net++ architecture" height="480">
</div>
The LAEO-Net++ receives as input two tracks of head crops and a tracks of maps containing the relative position of the heads, and 
returns the probability of being LAEO those two heads.



### Quick start

The following demo predicts the LAEO label on a pair of heads included in 
subdirectory `data/ava_val_crop`. You can choose either to use a model trained on UCO-LAEO 
or a model trained on AVA-LAEO.   

```bash
cd laeonetplus
python mains/ln_demo_test.py
```

*Training code will be available soon.*

### Detecting and tracking heads

See older [GitHub repository](https://github.com/AVAuco/laeonet).

### Install

Clone this repository:
```bash
git clone git@github.com:AVAuco/laeonet.git
```

### References
Marin-Jimenez, M. J., Kalogeiton, V., Medina-Suarez, P., Zisserman, A. (2021). [LAEO-Net++: revisiting people Looking At Each Other in videos.](https://www.researchgate.net/profile/Manuel_Marin-Jimenez/publication/347975905_LAEO-Net_revisiting_people_Looking_At_Each_Other_in_videos/links/5feb137592851c13fed05037/LAEO-Net-revisiting-people-Looking-At-Each-Other-in-videos.pdf) IEEE transactions on Pattern Analysis and Machine Intelligence, PP, 10.1109/TPAMI.2020.3048482. Advance online publication. https://doi.org/10.1109/TPAMI.2020.3048482

```
@article{marin21pami,
  author    = {Mar\'in-Jim\'enez, Manuel J. and Kalogeiton, Vicky and Medina-Su\'arez, Pablo and and Zisserman, Andrew},
  title     = {{LAEO-Net++}: revisiting people {Looking At Each Other} in videos},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  doi       = {10.1109/TPAMI.2020.3048482},
  year      = {2021}
}
```

Conference version:
```
@inproceedings{marin19cvpr,
  author    = {Mar\'in-Jim\'enez, Manuel J. and Kalogeiton, Vicky and Medina-Su\'arez, Pablo and and Zisserman, Andrew},
  title     = {{LAEO-Net}: revisiting people {Looking At Each Other} in videos},
  booktitle = CVPR,
  year      = {2019}
}
```