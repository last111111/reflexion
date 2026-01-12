## Meta-learning for Task-oriented Household Text Games

This code repository is modified from [alfworld](https://github.com/alfworld/alfworld).

The pre-trained model for reproducing the experiment results can be downloaded at [here](https://drive.google.com/file/d/1VX-wUlzGFYflA_CIo109zneXlBb-vPgB/view?usp=sharing).

After navigating to `./scripts`, we can train and evaluate the model with the following scripts. 

#### train

`python train_dqn.py`

#### evaluate

`python adapt_dqn.py`

## References

Marc-Alexandre C{\^o}t{\'e}, {\'A}kos K{\'a}d{\'a}r, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. 2018. Textworld: A learning environment for text-based games. In Computer Games Workshop at ICML/IJCAI 2018.

Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. 2020. ALFRED: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10740–10749.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre C{\^o}t{\'e}, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings of the International Conference on Learning Representations (ICLR).

Rasool Fakoor, Pratik Chaudhari, Stefano Soatto, and Alexander J. Smola. 2020. Meta-Q-learning. In Proceedings of the International Conference on Learning Representations (ICLR).

Elad Hazan, Sham Kakade, Karan Singh, and Abby Van Soest. 2019. Provably efficient maximum entropy exploration. In Proceedings of the 36th International Conference on Machine Learning (ICML).