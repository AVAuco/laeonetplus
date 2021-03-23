# LAEO-Net++
  

[![Try In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11abr3TV6Nb3pbjVTzE_tUOvMKRdE2NRm?usp=sharing)


Official support code for [LAEO-Net++ paper](https://arxiv.org/abs/2101.02136) (IEEE TPAMI, 2021).

<div align="center">
    <img src="./LAEOnetplus.png" alt="The LAEO-Net++ architecture" height="480">
</div>
The LAEO-Net++ receives as input two tracks of head crops and a track of maps containing the relative position of the heads, and 
returns the probability of being LAEO those two heads.



See previous version [here](https://github.com/AVAuco/laeonet)

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

Try this Google Colab to run LAEO prediction on a target video file: [notebook](https://colab.research.google.com/drive/1H28dEUORUmKIJGSh6N3B5sa_ocQsT1mT?usp=sharing)


### Install

Clone this repository:
```bash
git clone git@github.com:AVAuco/laeonetplus.git
```

See library dependencies [here](doc/dependencies.md). In addition, a [requirements.txt file](requirements.txt) 
is provided.

### Training (_in progress_)

We provide an example of training on AVA-LAEO data (using preprocessed samples): see the [training document](doc/training.md).


### References
Marin-Jimenez, M. J., Kalogeiton, V., Medina-Suarez, P., Zisserman, A. (2021). [LAEO-Net++: revisiting people Looking At Each Other in videos.](https://www.researchgate.net/profile/Manuel_Marin-Jimenez/publication/347975905_LAEO-Net_revisiting_people_Looking_At_Each_Other_in_videos/links/5feb137592851c13fed05037/LAEO-Net-revisiting-people-Looking-At-Each-Other-in-videos.pdf) IEEE transactions on Pattern Analysis and Machine Intelligence, PP, 10.1109/TPAMI.2020.3048482. Advance online publication. https://doi.org/10.1109/TPAMI.2020.3048482

```bibtex
@article{marin21pami,
  author    = {Mar\'in-Jim\'enez, Manuel J. and Kalogeiton, Vicky and Medina-Su\'arez, Pablo and and Zisserman, Andrew},
  title     = {{LAEO-Net++}: revisiting people {Looking At Each Other} in videos},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  doi       = {10.1109/TPAMI.2020.3048482},
  year      = {2021}
}
```

Conference version:
```bibtex
@inproceedings{marin19cvpr,
  author    = {Mar\'in-Jim\'enez, Manuel J. and Kalogeiton, Vicky and Medina-Su\'arez, Pablo and and Zisserman, Andrew},
  title     = {{LAEO-Net}: revisiting people {Looking At Each Other} in videos},
  booktitle = CVPR,
  year      = {2019}
}
```
