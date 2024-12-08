# Anatomic Boundary-aware Explanation for Convolutional Neural Networks in Diagnostic Radiology
### Citation
* H. Yuan. Anatomic Boundary‐aware Explanation for Convolutional Neural Networks in Diagnostic Radiology. iRadiology (2024): 1-16. https://doi.org/10.1002/ird3.113
* H. Yuan, C. Hong, PT. Jiang, G. Zhao, N. T. A. Tran, X. Xu, et al. Clinical Domain Knowledge-Derived Template Improves Post Hoc AI Explanations in Pneumothorax Classification. Journal of Biomedical Informatics (2024): 104673. https://doi.org/10.1016/j.jbi.2024.104673
* H. Yuan, C. Hong, N. T. A. Tran, X. Xu, and N. Liu. Leveraging Anatomical Constraints with Uncertainty for Pneumothorax Segmentation. Health Care Science (2024): 1-18. https://doi.org/10.1002/hcs2.119

> ### Please read our [article](https://) for further information.
We propose a plug-and-play module that explicitly integrates anatomic boundary information into the explanation process for CNN-based thoracopathy classification. To generate the anatomic boundary of the lung parenchyma, we utilize a lung segmentation model developed on external public datasets and deploy it on the unseen target dataset for CNN explanations in the clinical task of thoracopathy classification.

As outlined in Figure 1, the proposed method develops an auxiliary lung segmenter based on the external lung segmentation dataset. Upon the completion of segmenter training, each chest radiograph from the unseen target dataset was supplemented with a boundary image. This boundary image constrained the baseline focus area within the predicted lung region and improved the quality of model explanations through a plug-and-play approach. 

![](https://github.com/Han-Yuan-Med/constrained-explanation/blob/main/Constrained%20explanation-pipeline.png)
*Figure 1: Schematic diagram of the proposed boundary-aware explanation*
