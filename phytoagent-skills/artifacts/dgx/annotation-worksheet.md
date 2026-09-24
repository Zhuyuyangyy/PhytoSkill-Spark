# 标注表

每行一张图。**先自己看图**，再对照模型预测，填最后一列。

`annotator_labels` 用候选集里的词（可多选）；模型漏报的你也填进去——
漏检和误报要分开记，只记一个会高估准确率。

## 药材切片（15 张）

候选性状：`cut_surface_fissure` 断面裂隙/炸裂；`cut_surface_powder` 断面粉性；`cut_surface_dense` 断面致密/角质；`cut_surface_hollow` 断面空心；`colour_pale_yellow` 色泽淡黄（硫熏后常见）；`colour_amber` 色泽黄棕/琥珀；`colour_dark_brown` 色泽深褐/焦褐；`mould_visible` 可见霉斑；`insect_damage` 虫蛀孔道；`slice_irregular` 片型不整/厚薄不均

| 图片（文件名） | 尺寸 | 模型预测 | 观察数 | 你认可的性状（待填） | 模型对不对（待填） |
| --- | --- | --- | --- | --- | --- |
| `huangqi_01.jpg` | 800x800 | colour_pale_yellow、colour_amber、cut_surface_dense | 6 |  |  |
| `huangqi_02.jpg` | 800x800 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_03.jpg` | 800x533 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_04.jpg` | 1200x800 | colour_pale_yellow | 19 |  |  |
| `huangqi_05.jpg` | 3120x3120 | colour_pale_yellow、cut_surface_dense、slice_irregular | 3 |  |  |
| `huangqi_06.jpg` | 800x800 | cut_surface_dense、colour_pale_yellow、colour_dark_brown | 15 |  |  |
| `huangqi_07.jpg` | 800x800 | colour_pale_yellow | 1 |  |  |
| `huangqi_08.jpg` | 1500x1500 | colour_pale_yellow、cut_surface_dense、slice_irregular | 3 |  |  |
| `huangqi_09.jpg` | 800x800 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_10.jpg` | 599x400 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_11.jpg` | 800x533 | colour_pale_yellow | 8 |  |  |
| `huangqi_12.jpg` | 800x625 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_13.jpg` | 800x533 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_14.jpg` | 800x800 | colour_pale_yellow、cut_surface_dense | 21 |  |  |
| `huangqi_15.jpg` | 700x1053 | （模型未报告任何区域） | 0 |  |  |

## 植株叶片（15 张）

候选性状：`leaf_yellowing` 叶片黄化；`leaf_spot` 叶片斑点；`wilting` 萎蔫

| 图片（文件名） | 尺寸 | 模型预测 | 观察数 | 你认可的性状（待填） | 模型对不对（待填） |
| --- | --- | --- | --- | --- | --- |
| `huangqi_leaf_01.jpg` | 478x336 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_02.jpg` | 1920x2560 | leaf_yellowing | 1 |  |  |
| `huangqi_leaf_03.jpg` | 1616x1152 | leaf_yellowing | 1 |  |  |
| `huangqi_leaf_04.jpg` | 1024x768 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_05.jpg` | 600x400 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_06.jpg` | 445x315 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_07.jpg` | 751x500 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_08.jpg` | 750x500 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_09.jpg` | 406x612 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_10.jpg` | 500x620 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_11.jpg` | 500x375 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_12.jpg` | 750x500 | leaf_yellowing | 1 |  |  |
| `huangqi_leaf_13.jpg` | 475x633 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_14.jpg` | 612x406 | （模型未报告任何区域） | 0 |  |  |
| `huangqi_leaf_15.jpg` | 1024x682 | （模型未报告任何区域） | 0 |  |  |

## 填完之后

把结果写回 `artifacts/dgx/annotation-worksheet.json` 的 `annotator_labels` 和
`annotator_agrees`，或直接告诉我，我用 `dgx/score_annotations.py` 算：

- precision：模型报的性状里，有多少你也认可
- recall：你认可的性状里，有多少模型也报了
- 图像可用性一致率：模型说不可用 vs 你说不可用

只报 precision 会高估（模型可以少报），只报 recall 也会高估（模型可以乱报），
所以两个都要。
