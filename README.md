# Visual Classifier Baselines <br>
This model include 7 basline methods and some [datasets](https://pan.baidu.com/s/1zEDDzRB2Dbz_otDWUMflMQ).<br>

## **1. Sparse Representation based Classifier ([SRC](https://www.ideals.illinois.edu/bitstream/handle/2142/103886/08-2203.pdf?sequence=2&isAllowed=y))**
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2008<br>
This method has been optimized by two approaches,<br>
i) Alternating Direction Method of Multipliers (ADMM), the main functiona is **main_SRC_ADMM**<br>
ii) Blockwise Coordinate Descent (BCD), the main functiona is **main_SRC**

## **2. Collaborative Representation based Classifier ([CRC](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.231.8008&rep=rep1&type=pdf))**
IEEE International Conference on Computer Vision (ICCV), 2011<br>
The main functiona is **main_CRC**

## **3. Nonnegative Representation based Classifier ([NRC](https://arxiv.org/pdf/1806.04329.pdf))**
Pattern Recognition (PR), 2019<br>
The main functiona is **main_NRC**

## **4. Superposed Linear Representation Classifier ([SLRC](http://whdeng.cn/papers/18_Deng_PAMI.pdf))**
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2018<br>
The main functiona is **main_SLRC**

## **5. Euler Sparse Representation based Classifier ([Euler-SRC](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16524/16587))**
AAAI Conference on Artificial Intelligence (AAAI), 2018<br>
The main functiona is **main_Euler_SRC**

## **6. Label Consistent K-SVD ([LC-KSVD](https://cyber.sci-hub.se/MTAuMTEwOS90cGFtaS4yMDEzLjg4/zhuolinjiang2013.pdf#view=FitH))**
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2013<br>
The main functiona is **main_LCKSVD**

## **7. Class Specific Dictionary Learning ([CSDL](https://dacemirror.sci-hub.se/journal-article/cd86b39431cd2ca2aa07e2e30a2ca51e/liu2016.pdf#view=FitH))**
Neurocomputing , 2016<br>
The main functiona is **main_CSDL**

## License
Our code is released under MIT License (see LICENSE file for details).
