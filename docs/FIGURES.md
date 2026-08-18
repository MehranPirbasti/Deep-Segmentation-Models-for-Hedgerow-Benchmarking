# Figure list and captions

Figure numbers follow the order of appearance in the manuscript. Each entry gives
the file(s), the data table the figure is drawn from, and the caption as it
appears in the paper.

Five figures contain Pleiades Neo imagery and are therefore **not** published
here; they are available in the article. They are marked *withheld* below, and
the reason is given in `figures/png/README.md`. No table, figure datum or
reported number depends on them.

| # | Label | File(s) | Data | Vector |
|---|---|---|---|---|
| 1 | `fig:benchmark_workflow` | `Workflow.png` | - | no |
| 2 | `fig:study_area` | `Study_Area.jpg` | *withheld, see below* | no |
| 3 | `fig:proxy_annotation_update` | `Proxy_Annotation_Update.jpg` | *withheld, see below* | no |
| 4 | `fig:radar` | `Test_Set_Radar_All.png` | Test_Set_Radar_All.csv | yes |
| 5 | `fig:leaderboard_matrix` | `Leaderboard_Matrix_Main.png` | Leaderboard_Matrix_Main.csv | yes |
| 6 | `fig:heatmap` | `Leaderboard_Matrix_Hybrid.png` | Leaderboard_Matrix_Hybrid.csv | yes |
| 7 | `fig:learning_curves` | `Learning_Curves_IoU.png` | Learning_Curves_IoU.csv | yes |
| 8 | `fig:multiseed_boxplot` | `MultiSeed_CI_IoU.png` | MultiSeed_CI_IoU.csv | yes |
| 9 | `fig:topology_matrix` | `Topology_Matrix.png` | Topology_Matrix.csv | yes |
| 10 | `fig:bf1_tolerance_fig` | `BF1_Tolerance.png` | BF1_Tolerance.csv | yes |
| 11 | `fig:decoder_boxplot` | `Decoder_Boxplot.png` | Decoder_Boxplot.csv | yes |
| 12 | `fig:backbone_boxplots` | `Boxplot_boundary_f1.png`, `Boxplot_dice_f1.png`, `Boxplot_iou.png`, `Legend_Key.png` | Boxplot_boundary_f1.csv, Boxplot_dice_f1.csv, Boxplot_iou.csv | yes |
| 13 | `fig:efficiency_frontier` | `Efficiency_Frontier_boundary_f1.png`, `Efficiency_Frontier_dice_f1.png`, `Efficiency_Frontier_iou.png`, `Legend_Key.png` | Efficiency_Frontier_boundary_f1.csv, Efficiency_Frontier_dice_f1.csv, Efficiency_Frontier_iou.csv | yes |
| 14 | `fig:training_cost` | `Training_Efficiency_boundary_f1.png`, `Training_Efficiency_dice_f1.png`, `Training_Efficiency_iou.png`, `Legend_Key.png` | Training_Efficiency_boundary_f1.csv, Training_Efficiency_dice_f1.csv, Training_Efficiency_iou.csv | yes |
| 15 | `fig:quantitative_overlays` | `Quantitative_Overlays.png` | *withheld, see below* | no |
| 16 | `fig:qualitative_overlays` | `Qualitative_Overlays.jpg` | *withheld, see below* | no |
| 17 | `fig:heterogeneous_examples` | `Heterogeneous_Conditions_Panel.png` | *withheld, see below* | no |
| 18 | `fig:encoder_frontier` | `Encoder_Family_Frontier.png` | Encoder_Family_Frontier.csv | yes |
| 19 | `fig:topology_indices` | `Topology_Indices.png` | Topology_Indices.csv | yes |
| 20 | `fig:topology_frontier` | `Topology_Frontier.png` | Topology_Frontier.csv | yes |

---

### Figure 1, `fig:benchmark_workflow`

**Files:** `figures/png/Workflow.png`  
**Vector:** not applicable (photograph or diagram)  
**Data:** -

**Caption:** Overview of the benchmark workflow, including data preparation, patch splitting, representative model taxonomy, and evaluation outputs used in the reported experiments. Panel(c) presents illustrative examples of decoder types and backbone families, rather than an exhaustive listing of all model configurations.

### Figure 2, `fig:study_area`

**Files:** *withheld (contains licensed imagery); see the article*  
**Vector:** not applicable (photograph or diagram)  
**Data:**, (imagery)

**Caption:** The right panel shows our study area located in the UK. The left panel shows the outline of the HR satellite imagery. Imagery (c) Airbus DS 2022.

### Figure 3, `fig:proxy_annotation_update`

**Files:** *withheld (contains licensed imagery); see the article*  
**Vector:** not applicable (photograph or diagram)  
**Data:**, (imagery)

**Caption:** Annotated layer on UKCE proxy map: The red LiDAR-proxy data is complemented by an updated blue annotation layer. The image shows that some hedgerows are missing due to newer imagery, while new features have been added. Imagery (c) Airbus DS 2022; reference layer (c) UKCEH.

### Figure 4, `fig:radar`

**Files:** `figures/png/Test_Set_Radar_All.png`  
**Vector:** Test_Set_Radar_All.svg  
**Data:** Test_Set_Radar_All.csv

**Caption:** Normalized multi-axis comparison of every evaluated configuration, split into (a) matched and (b) hybrid decoder-backbone pairings. The five axes are IoU, Dice/F1, BF1 at r=2, and the inverted inference and training costs, each min-max normalized over the full set so that a larger enclosed area is better on every axis. Color identifies the decoder family and the marker identifies the encoder backbone, using the same convention throughout the paper; solid lines denote convolutional encoders and dashed lines transformer encoders.

### Figure 5, `fig:leaderboard_matrix`

**Files:** `figures/png/Leaderboard_Matrix_Main.png`  
**Vector:** Leaderboard_Matrix_Main.svg  
**Data:** Leaderboard_Matrix_Main.csv

**Caption:** Leaderboard matrix summarizing test-set operating points across the main group model variants.

### Figure 6, `fig:heatmap`

**Files:** `figures/png/Leaderboard_Matrix_Hybrid.png`  
**Vector:** Leaderboard_Matrix_Hybrid.svg  
**Data:** Leaderboard_Matrix_Hybrid.csv

**Caption:** Leaderboard matrix summarizing test-set operating points across the hybrid group model variants.

### Figure 7, `fig:learning_curves`

**Files:** `figures/png/Learning_Curves_IoU.png`  
**Vector:** Learning_Curves_IoU.svg  
**Data:** Learning_Curves_IoU.csv

**Caption:** Validation convergence by encoder family. The line is the median across decoders and the band spans the full range, for each of the six encoder backbones; panel (b) is a detail of the plateau. Plotting per-family envelopes rather than one line per configuration keeps the comparison legible at column width while showing the same trajectories.

### Figure 8, `fig:multiseed_boxplot`

**Files:** `figures/png/MultiSeed_CI_IoU.png`  
**Vector:** MultiSeed_CI_IoU.svg  
**Data:** MultiSeed_CI_IoU.csv

**Caption:** Multi-seed test IoU under the frozen protocol: mean (diamond) and 95% confidence interval (whiskers) over five seeds per configuration. The shaded band marks the gap between the lower bound of U-Net++ (ResNet-50) and the highest upper bound of the remaining configurations; because the band is non-empty, that advantage survives seed variation. The second-tier intervals overlap one another and are therefore read as statistically indistinguishable rather than as an ordering.

### Figure 9, `fig:topology_matrix`

**Files:** `figures/png/Topology_Matrix.png`  
**Vector:** Topology_Matrix.svg  
**Data:** Topology_Matrix.csv

**Caption:** Topology-aware evaluation matrix on the test set. Cells are color-coded by relative performance (green better) with raw values annotated; the Betti-0 error and fragmentation-index columns are inverted for coloring so that green consistently denotes better connectivity. Expansive models (LinkNet, FPN) combine low fragmentation with high Betti-0 error, the signature of false bridging.

### Figure 10, `fig:bf1_tolerance_fig`

**Files:** `figures/png/BF1_Tolerance.png`  
**Vector:** BF1_Tolerance.svg  
**Data:** BF1_Tolerance.csv

**Caption:** BF1 at the two boundary tolerances. The bar length is the penalty incurred by tightening the tolerance from r=2 to r=1; expansive, boundary-thickening configurations lose roughly twice as much as geometrically precise ones.

### Figure 11, `fig:decoder_boxplot`

**Files:** `figures/png/Decoder_Boxplot.png`  
**Vector:** Decoder_Boxplot.svg  
**Data:** Decoder_Boxplot.csv

**Caption:** Distribution of best validation IoU across decoder families over the evaluated backbone variants.

### Figure 12, `fig:backbone_boxplots`

**Files:** `figures/png/Boxplot_boundary_f1.png`, `figures/png/Boxplot_dice_f1.png`, `figures/png/Boxplot_iou.png`, `figures/png/Legend_Key.png`  
**Vector:** Boxplot_boundary_f1.svg, Boxplot_dice_f1.svg, Boxplot_iou.svg, Legend_Key.svg  
**Data:** Boxplot_boundary_f1.csv, Boxplot_dice_f1.csv, Boxplot_iou.csv

**Caption:** Encoder-family effect across every evaluated configuration: distribution of test BF1, Dice/F1 and IoU by backbone, with every configuration shown as an individual point (n per backbone is annotated). Color identifies the decoder family and the marker the encoder backbone, as given in the key at the foot of the figure.

### Figure 13, `fig:efficiency_frontier`

**Files:** `figures/png/Efficiency_Frontier_boundary_f1.png`, `figures/png/Efficiency_Frontier_dice_f1.png`, `figures/png/Efficiency_Frontier_iou.png`, `figures/png/Legend_Key.png`  
**Vector:** Efficiency_Frontier_boundary_f1.svg, Efficiency_Frontier_dice_f1.svg, Efficiency_Frontier_iou.svg, Legend_Key.svg  
**Data:** Efficiency_Frontier_boundary_f1.csv, Efficiency_Frontier_dice_f1.csv, Efficiency_Frontier_iou.csv

**Caption:** Efficiency frontier on the test set for every evaluated configuration: BF1, Dice/F1 and IoU against per-patch inference time. The dashed line is the Pareto frontier. Color identifies the decoder family and the marker the encoder backbone, as given in the key at the foot of the figure.

### Figure 14, `fig:training_cost`

**Files:** `figures/png/Training_Efficiency_boundary_f1.png`, `figures/png/Training_Efficiency_dice_f1.png`, `figures/png/Training_Efficiency_iou.png`, `figures/png/Legend_Key.png`  
**Vector:** Training_Efficiency_boundary_f1.svg, Training_Efficiency_dice_f1.svg, Training_Efficiency_iou.svg, Legend_Key.svg  
**Data:** Training_Efficiency_boundary_f1.csv, Training_Efficiency_dice_f1.csv, Training_Efficiency_iou.csv

**Caption:** Training cost trade-offs on the test set: accuracy metrics versus training duration.

### Figure 15, `fig:quantitative_overlays`

**Files:** *withheld (contains licensed imagery); see the article*  
**Vector:** not applicable (photograph or diagram)  
**Data:**, (imagery)

**Caption:** Representative quantitative overlays for selected model variants. The figure contrasts model predictions against the reference annotation on the same image regions, enabling visual comparison of continuity, fragmentation, false bridging, and boundary fidelity. Imagery (c) Airbus DS 2022.

### Figure 16, `fig:qualitative_overlays`

**Files:** *withheld (contains licensed imagery); see the article*  
**Vector:** not applicable (photograph or diagram)  
**Data:**, (imagery)

**Caption:** Representative qualitative overlays for the selected model variants. Each panel shows prediction (green) and reference annotation (blue) on the same image region. Imagery (c) Airbus DS 2022.

### Figure 17, `fig:heterogeneous_examples`

**Files:** *withheld (contains licensed imagery); see the article*  
**Vector:** not applicable (photograph or diagram)  
**Data:**, (imagery)

**Caption:** Heterogeneous hedgerow conditions in the study area. Each row shows a test sample, its reference annotation, and predictions for the three behavior patterns identified in Section Table qualitative: conservative (U-Net++ (ResNet-50)), balanced (UPerNet (MiT-B4)), and expansive (LinkNet (ResNet-34)). Test samples are displayed with a common radiometric normalization so the conditions are comparable; the irregular, unique-form condition is not repeated here because that patch is the one shown in Fig. Fig. 15; black areas fall outside the acquisition footprint. The prediction masks are synthesized for illustration and are not measured outputs; quantitative results are reported in Tables Table best_by_arch_main_hybrid and Table two_way_best_summary. Imagery (c) Airbus DS 2022.

### Figure 18, `fig:encoder_frontier`

**Files:** `figures/png/Encoder_Family_Frontier.png`  
**Vector:** Encoder_Family_Frontier.svg  
**Data:** Encoder_Family_Frontier.csv

**Caption:** Encoder-family effect on the accuracy-cost frontier, for (a) region overlap and (b) centerline agreement. Each family is summarized by its median cost and median score with interquartile bars on both axes; the faint dots are the individual configurations. The two panels order the families differently: ResNet-34 and ResNet-50 lead on IoU at low cost, whereas MiT-B4 and MiT-B2 lead on clDice, so the ordering that holds for area does not hold for connectivity.

### Figure 19, `fig:topology_indices`

**Files:** `figures/png/Topology_Indices.png`  
**Vector:** Topology_Indices.svg  
**Data:** Topology_Indices.csv

**Caption:** The three topology-aware indices against region overlap, over every evaluated configuration: (a) clDice, higher is better; (b) normalized Betti-0 error and (c) fragmentation index, lower is better. Color identifies the encoder backbone and marker shape the decoder; a ringed marker denotes a configuration measured in Table Table topology, the remainder being extended from those measurements. The dashed line and the quoted rank correlation are computed on the measured rows alone, since the extended values are derived from accuracy and including them would report an association the extension itself created. So computed, clDice tracks IoU only loosely (=+0.43) whereas both error indices decline with it.

### Figure 20, `fig:topology_frontier`

**Files:** `figures/png/Topology_Frontier.png`  
**Vector:** Topology_Frontier.svg  
**Data:** Topology_Frontier.csv

**Caption:** Connectivity against inference cost over every evaluated configuration: (a) clDice, higher is better, and (b) normalized Betti-0 error, lower is better. The Pareto frontier is drawn in the improving direction of each index; ringed markers denote configurations measured in Table Table topology. Neither frontier improves beyond about 3 s, so the cost of the deepest encoders buys no connectivity either.
