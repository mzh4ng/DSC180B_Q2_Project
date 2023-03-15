---
title: Introduction
author: Benjamin Sacks
date: 2000-01-01
category: Jekyll
layout: post
cover: /DSC180B_Q2_Project/assets/microbiome.png
---

Each year, an estimated two million Americans receive a cancer diagnosis. Patient characteristics such as age, gender, and general health status can impact cancer progression and response to treatment modalities like chemotherapy. Nonetheless, a crucial yet often overlooked element that may hold significant sway is the patient's microbiome. While humans possess approximately 20,000 genes in our DNA, we also harbor a substantial number of microbial genes, ranging from 2 to 20 million throughout our various bodily microbiomes. 

{:refdef: style="text-align: center;"}
![Comparing Human Gene Count to Microbial Count](/DSC180B_Q2_Project/assets/human-vs-microbes.png){:class="img-responsive"}{: width="400" }
{: refdef}

Furthermore, despite a 99.99% DNA similarity between two strangers, their gut microbiomes may only share 10% similarity. In numerous instances, the microbiome composition dictates medication efficacy and disease susceptibility. For example, one study investigated the effectiveness of Cordyceps militaris extract in overcoming carboplatin resistance in ovarian cancer and found that the extract reduced the viability of carboplatin-resistant SKOV-3 cells and induced apoptosis. (Jo et al.) 

{:refdef: style="text-align: center;"}
![Comparing Human Gene Count to Microbial Count](https://cdn.shopify.com/s/files/1/0514/0101/products/2008-12-14_Cordyceps_militaris_3107128906.jpg?v=1551741041){:class="img-responsive"}{: width="400" }
{: refdef}

Consequently, it is plausible that mycobiomes might partly contribute to the differential cancer progression rates observed in some individuals.

## Literature Review and Discussion of Prior Work

In the past, researchers have found that bacteria microbes were present in over 1500 tumors spanning seven types of cancer (Nejman et al). The study identified both cancer cells and immune cells as being sites for microbiomes, and that the bacterial composition varied by cancer type. Following this, researchers at the University of California, San Diego re-examined sequencing studies in The Cancer Genome Atlas (TCGA) of 33 types of cancer from treatment-naive patients (a total of 18,116 samples) for microbial reads (Poore et al). They found that they could diagnose cancer type in individuals with stage Ia–IIc cancer and cancers lacking any genomic alterations. 

![Comparing Human Gene Count to Microbial Count](/DSC180B_Q2_Project/assets/poore.jpg){:class="img-responsive"}

Furthermore, they were able to distinguish between healthy individuals and individuals with multiple cancers solely using microbial signatures. Additionally, a paper published earlier this year also found that multi-kingdom microbiota was effective at diagnosing colorectal cancer (Liu et al). 

The study that we based our research off of for this project was the pan-cancer analysis which revealed cancer-type-specific fungal ecologies (Poore et al.). In this study mycobial data sourced from TCGA was used to distinguish between multiple types of cancers tumors. However, this study had intentionally left out metadata from the analysis in order to emphasize mycobial community impact. Our goal for this project was to see if the reintroduction of the metadata could have a positive impact on identifying pathological stage and the days between diagnosis and death for individuals with various cancers.

{:refdef: style="text-align: center;"}
![Comparing Human Gene Count to Microbial Count](https://www.uab.edu/news/media/k2/items/cache/e84ec87f15dcc0ea491d4bb9e6b133bd_XL.jpg){:class="img-responsive"}{: height="200"}
![Comparing Human Gene Count to Microbial Count](https://tasteforlife.com/sites/default/files/styles/facebook/public/conditions-wellness/digestion/meet-your-mycobiome/meet-your-mycobiome.jpg?itok=dsf3ktmf){:class="img-responsive"}{: height="200"}
{: refdef}
