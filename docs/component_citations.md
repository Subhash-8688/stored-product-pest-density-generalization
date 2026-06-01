# Component Citations and Attribution

This repository adapts or integrates established methods within a YOLOv11s detection pipeline. Cite the original method papers when describing each component in a manuscript. A ready-to-import BibTeX file is available at [`references.bib`](references.bib).

## Required Component Citations

| Repository component | Manuscript citation | Where to cite it |
| --- | --- | --- |
| YOLOv11s baseline and Ultralytics training framework | Jocher and Qiu (2024) | First introduction of the YOLOv11s baseline and the experimental setup |
| Ghost Convolution (GC) | Han et al. (2020) | First description of lightweight backbone feature extraction |
| DySample (DS) | Liu et al. (2023) | First description of dynamic neck upsampling |
| Focal-EIoU (FE) | Zhang et al. (2022) | Bounding-box regression loss subsection |
| Efficient Multi-Scale Attention (EMA) | Ouyang et al. (2023) | Separate attention-mechanism study |
| Convolutional Block Attention Module (CBAM) | Woo et al. (2018) | Separate attention-mechanism study |
| Efficient Channel Attention (ECA) | Wang et al. (2020) | Separate attention-mechanism study |
| Standalone Channel Attention | Woo et al. (2018) | Separate attention-mechanism study; this variant uses the channel-refinement component exposed by the CBAM implementation |

## Supporting Method Citations

These references are required when the corresponding preprocessing, audit, or visualization method is described in the manuscript.

| Repository method | Manuscript citation | Where to cite it |
| --- | --- | --- |
| Grad-CAM visualization | Selvaraju et al. (2017) | Qualitative feature-activation visualization subsection |
| Structural Similarity Index Measure (SSIM) | Wang et al. (2004) | Cross-split leakage audit subsection when SSIM is reported |
| MixUp augmentation | Zhang et al. (2018) | Data augmentation subsection |
| RandAugment | Cubuk et al. (2020) | Data augmentation subsection |
| Mosaic augmentation | Bochkovskiy et al. (2020) | Data augmentation subsection |

## Suggested Manuscript Wording

For the principal architectural modifications:

> The YOLOv11s baseline was modified by integrating Ghost Convolution (GC) for lightweight feature extraction (Han et al., 2020), DySample (DS) for point-sampling-based dynamic upsampling in the neck (Liu et al., 2023), and Focal-EIoU (FE) for bounding-box regression with effective example mining (Zhang et al., 2022).

For the separate attention study:

> To evaluate semantic feature recalibration after the C2PSA block, four attention variants were investigated: Efficient Multi-Scale Attention (EMA) (Ouyang et al., 2023), Convolutional Block Attention Module (CBAM) (Woo et al., 2018), Efficient Channel Attention (ECA) (Wang et al., 2020), and the standalone channel-attention component associated with CBAM (Woo et al., 2018).

## Original Papers and Code Resources

- YOLO11 documentation and software citation: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- GhostNet paper: [CVPR 2020 open-access page](https://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html)
- GhostNet code resource: [huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet)
- DySample paper: [ICCV 2023 open-access page](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html)
- DySample reference implementation: [tiny-smart/dysample](https://github.com/tiny-smart/dysample)
- Focal-EIoU paper: [Neurocomputing article](https://doi.org/10.1016/j.neucom.2022.07.042)
- EMA paper: [IEEE Xplore](https://doi.org/10.1109/ICASSP49357.2023.10096516)
- CBAM paper: [ECCV 2018 open-access page](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)
- ECA-Net paper: [CVPR 2020 open-access page](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html)
- ECA-Net reference implementation: [BangguWu/ECANet](https://github.com/BangguWu/ECANet)
- Grad-CAM paper: [ICCV 2017 open-access paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)
- SSIM paper: [IEEE Xplore article](https://doi.org/10.1109/TIP.2003.819861)
- MixUp paper: [ICLR 2018 OpenReview page](https://openreview.net/forum?id=r1Ddp1-Rb)
- RandAugment paper: [CVPR Workshops 2020 open-access page](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)
- Mosaic augmentation source: [YOLOv4 arXiv paper](https://arxiv.org/abs/2004.10934)

## Implementation Note

The repository does not redistribute the original external repositories. It provides an Ultralytics patch containing the chapter-specific integration. The attribution above distinguishes the original method contributions from this study's integration, placement choices, density-stratified evaluation protocol, and pest-detection experiments.
