<new_puzzle>
<description>
网格中由分隔线围成若干面板，每个面板内有一个小图案。相邻面板如果图案完全相同，则它们之间的分隔线段变为Green；如果图案不同，分隔线段变为Red。外框保持不变。

Input structure:
- 背景为Black。
- Orange像素组成完整的水平线和垂直线，将网格均匀分割为若干大小相同的矩形面板（例如2x3个面板排列）。
- 每个面板内部包含由Blue像素在Black背景上绘制的小型图案。
- 部分面板图案完全相同（逐像素一致），部分不同。
- 网格最外一圈由Orange像素围成边框。

Rule:
- 逐一检查每对在水平或垂直方向上直接相邻的面板。
- 如果两个相邻面板的内部图案逐像素完全相同，则将它们之间那段内部Orange分隔线（不含外框和交叉点处的像素）替换为Green。
- 如果有任何一个像素不同，则将该段分隔线替换为Red。
- 网格最外圈的Orange边框保持Orange不变。
- 分隔线交叉点处的像素（同时属于水平和垂直分隔线的像素）保持Orange不变。
- 面板内部图案本身不做任何改变。

Key insight:
做题者会本能地将注意力集中在面板内部图案上，试图找到图案变换规则（旋转、颜色替换等）。但实际上面板内容完全不变——变化发生在分隔线上，而分隔线通常被视为纯装饰性的结构元素。规则要求做题者从"关注面板内容"转向"关注面板间关系"，理解分隔线是"比较结果的显示器"。逐像素精确匹配的要求意味着微小差异（一个像素偏移）就会导致分隔线变Red，做题者需要精确比对面板内容。
</description>

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
O O O O O O O O O O
O B K K O B K K O K B K O
O K B K O K B K O K B K O
O O O O O O O O O O O O O
O B K K O K K B O K B K O
O K B K O K B K O K B K O
O O O O O O O O O O O O O
```

此网格为2行3列面板，每面板内部2x3。但尺寸不一致，重新设计：

Input（7行10列，2行3列面板，每面板内部2x2，分隔线宽1像素）：
```
O O O O O O O O O O
O B K O B K O K B O
O K B O K B O K B O
O O O O O O O O O O
O B K O K K O K B O
O K B O B K O K B O
O O O O O O O O O O
```

面板1(行1-2,列1-2): B K / K B
面板2(行1-2,列4-5): B K / K B → 与面板1相同
面板3(行1-2,列7-8): K B / K B → 与面板1不同
面板4(行4-5,列1-2): B K / K B → 与面板1相同
面板5(行4-5,列4-5): K K / B K → 与面板2不同
面板6(行4-5,列7-8): K B / K B → 与面板3相同

横向相邻关系：
面板1-面板2: 相同→分隔线列3行1-2变G
面板2-面板3: 不同→分隔线列6行1-2变R
面板4-面板5: 不同→分隔线列3行4-5变R
面板5-面板6: 不同→分隔线列6行4-5变R

纵向相邻关系：
面板1-面板4: 相同→分隔线行3列1-2变G
面板2-面板5: 不同→分隔线行3列4-5变R
面板3-面板6: 相同→分隔线行3列7-8变G

Output:
```
O O O O O O O O O O
O B K G B K R K B O
O K B G K B R K B O
O G G O R R O G G O
O B K R K K R K B O
O K B R B K R K B O
O O O O O O O O O O
```

说明：面板1和面板2图案相同，其间纵向分隔线(列3,行1-2)变Green。面板2和面板3不同，分隔线(列6,行1-2)变Red。面板1和面板4相同，横向分隔线(行3,列1-2)变Green。面板2和面板5不同，分隔线(行3,列4-5)变Red。面板3和面板6相同，分隔线(行3,列7-8)变Green。面板4和面板5不同，分隔线(列3,行4-5)变Red。面板5和面板6不同，分隔线(列6,行4-5)变Red。外框和交叉点保持Orange。
</new_puzzle>

<new_puzzle>
<description>
网格中有若干个彩色矩形物体和一条贯穿网格的Grey水平或垂直直线。每个矩形向Grey线投影，投影段染上该矩形的颜色；但当两个矩形的投影段在Grey线上重叠时，重叠区域互相抵消，保持Grey不变。

Input structure:
- 背景为Black。
- 一条Grey直线横贯或纵贯网格（占据完整的一行或一列）。
- 若干不同颜色的实心矩形物体分布在Grey线的两侧或同侧，矩形与Grey线不重叠。
- 矩形颜色互不相同，且不为Grey或Black。

Rule:
- 对每个矩形物体，正交投影到Grey线上：水平Grey线时投影为矩形列范围对应的线段；垂直Grey线时投影为矩形行范围对应的线段。
- 统计Grey线上每个像素被多少个矩形的投影覆盖：
  - 恰好被1个矩形投影覆盖→该像素变为该矩形颜色。
  - 被2个或更多矩形投影覆盖→该像素保持Grey（投影互相抵消）。
  - 未被任何投影覆盖→保持Grey。
- 矩形本身、背景和Grey线未被投影覆盖的部分均不变。

Key insight:
做题者看到Grey线上出现了彩色段，容易推测"矩形向线投影染色"的基本规则。但部分区域保持Grey会令人困惑——做题者会猜测是距离、大小等因素导致某些投影无效。真正的关键在于重叠抵消效应：两个矩形的投影在同一位置"中和"为Grey。发现这一点需要跨物体全局对比投影范围，而非逐个物体独立分析。
</description>

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K
K B B B K K K K
K B B B K K K K
E E E E E E E E
K K R R R R K K
K K R R R R K K
K K K K K K K K
```

Output:
```
K K K K K K K K
K B B B K K K K
K B B B K K K K
E B E E R R E E
K K R R R R K K
K K R R R R K K
K K K K K K K K
```

说明：Grey水平线在行3。Blue矩形(行1-2,列1-3)投影到列1-3。Red矩形(行4-5,列2-5)投影到列2-5。重叠区域=列2-3。行3各像素：列0无投影→E；列1仅Blue→B；列2-3被Blue和Red同时覆盖→抵消→保持E；列4-5仅Red→R；列6-7无投影→E。
</new_puzzle>

<new_puzzle>
<description>
网格中有若干个由不同颜色构成的矩形物体，每个物体旁边紧贴着一条Red标记线段。Red线段的位置指示物体被"挤压"的方向——物体沿标记所在边的方向塌缩为一条线，保留在原bounding box的该边上。

Input structure:
- 背景为Black。
- 若干个由不同单一颜色构成的实心矩形前景物体（至少2x2像素），彼此不重叠。
- 每个物体的外侧恰好有一条Red线段（1像素宽，长度等于物体在该边的跨度），紧贴物体的某一条边（上、下、左或右边外侧）。
- 不同物体的Red线段不会彼此接触。

Rule:
- 对每个物体，找到其Red标记线段贴在物体的哪一条边：
  - 贴在上边→物体向上挤压：物体塌缩为一条水平线，位于原bounding box最上行，宽度不变，其余行变Black。
  - 贴在下边→物体向下挤压：塌缩为bounding box最下行的水平线。
  - 贴在左边→向左挤压：塌缩为bounding box最左列的垂直线，高度不变。
  - 贴在右边→向右挤压：塌缩为bounding box最右列的垂直线。
- 物体被挤压后，原本占据的多余像素变为Black。
- 所有Red标记线段在输出中被移除（变为Black）。

Key insight:
做题者看到矩形旁的Red线段，第一反应是将其理解为方向指示器，预期物体沿该方向平移。但物体并没有移动，而是"塌缩"——一个维度的尺寸减少到1并停留在靠近Red标记的那一边。"挤压"概念不同于平移、旋转或镜像，做题者需要多个样例才能区分挤压与平移。Red线段的位置同时编码了塌缩轴和保留边两个信息。
</description>

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K
K R R R K K K K K
K B B B K K K K K
K B B B K K K K K
K B B B K K K K K
K K K K K K K K K
K K K K K G G G K
K K K K K G G G K
K K K K K G G G R
K K K K K G G G R
K K K K K G G G R
K K K K K K K K K
```

Output:
```
K K K K K K K K K
K K K K K K K K K
K B B B K K K K K
K K K K K K K K K
K K K K K K K K K
K K K K K K K K K
K K K K K K K G K
K K K K K K K G K
K K K K K K K G K
K K K K K K K G K
K K K K K K K G K
K K K K K K K K K
```

说明：Blue矩形(行2-4,列1-3)为3x3。Red线段在其上方(行1,列1-3)，紧贴上边→向上挤压。物体塌缩为bounding box最上行(行2,列1-3)的一条水平线"B B B"，行3-4变Black。Red标记删除。

Green矩形(行6-10,列5-7)为5x3。Red线段在其右侧(行8-10,列8)——但Red应等长于Green在该边的跨度。Green右边跨度为5行(行6-10)，Red只有3个像素。让我修正使Red线段长度等于物体在该边的跨度。

修正Input:
```
K K K K K K K K K
K R R R K K K K K
K B B B K K K K K
K B B B K K K K K
K B B B K K K K K
K K K K K G G G R
K K K K K G G G R
K K K K K G G G R
K K K K K K K K K
```

Blue矩形(行2-4,列1-3)为3x3，Red在上方(行1,列1-3)3像素→上边。
Green矩形(行5-7,列5-7)为3x3，Red在右侧(行5-7,列8)3像素→右边。

Output:
```
K K K K K K K K K
K K K K K K K K K
K B B B K K K K K
K K K K K K K K K
K K K K K K K K K
K K K K K K K G K
K K K K K K K G K
K K K K K K K G K
K K K K K K K K K
```

说明：Blue(3x3)上方有Red标记→向上挤压→塌缩为最上行(行2)的水平线"B B B"，行3-4变Black。Green(3x3)右侧有Red标记→向右挤压→塌缩为最右列(列7)的垂直线，行5-7列7为"G G G"，列5-6变Black。Red标记全部移除。
</new_puzzle>

<new_puzzle>
<description>
网格中有若干封闭的单色矩形环（框），每个框有且仅有一条边是缺失的（形成U形开口）。从开口方向向外发射一束与框同色的射线，射线的宽度等于开口宽度，射线的长度等于框内部空腔的行数或列数（取决于开口方向）。同时，网格中散布着若干与某些框同色的单像素"诱饵"，它们不参与任何变换。

Input structure:
- 背景为Black。
- 若干不完整矩形框（U形），每个由单一非Black颜色构成。每个框的四条边中恰好有一条完全缺失，形成一个开口。
- 框的颜色互不相同。
- 网格中另有若干孤立的单像素，颜色与某些框相同，作为干扰项。

Rule:
- 对每个U形框，确定缺失边的方向（上/下/左/右即开口方向）。
- 计算框内部空腔的尺寸：若开口在上或下，空腔行数=框内部高度，空腔列数=开口宽度；若开口在左或右，空腔列数=框内部宽度，空腔行数=开口高度。
- 从开口处向外（远离框内部的方向）发射射线：
  - 射线宽度=开口的宽度（即框内部在开口方向上的垂直跨度）。
  - 射线长度=空腔在开口方向上的深度（即框内部在开口方向上的跨度）。
- 射线颜色与框颜色相同。
- 射线不覆盖其他已有的非Black像素。
- 孤立单像素（诱饵）保持不变，不触发任何射线。

Key insight:
做题者看到同色的框和孤立像素，第一反应是建立框与像素之间的颜色匹配关系（移动、填充、连接等）。但孤立像素完全是干扰项——规则仅基于框本身的几何属性。做题者需要抑制"匹配同色对象"的直觉，意识到(1)孤立像素无关紧要，(2)框的缺失边编码了方向，(3)射线长度由框内部空腔尺寸决定。三个独立的认知步骤中任何一个缺失都无法正确推断规则。
</description>

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K K
K B B B B K K K K K
K B K K B K K K K K
K B K K B K K K K K
K K K K K K K G K K
K K K K K K K K K K
K K K K G G K K K K
K K K K G K G K K K
K K K K G K G K K K
K K K K G G G K K K
K K K K K K K K K K
```

Blue框分析(行1-3,列1-4)：
行1: B B B B（上边完整）
行2: B _ _ B（左右边各有B）
行3: B _ _ B（左右边各有B）
下边(行4): K K K K → 下边缺失，开口朝下。
左边(列1,行1-3): B B B ✓
右边(列4,行1-3): B B B ✓
内部空腔=行2-3,列2-3（2行2列）。空腔深度（开口方向=下，深度=空腔行数=2）。开口宽度=列2-3=2像素。
射线从开口处(行4,列2-3)向下发射，宽度2，长度2→填充(行4,列2-3)和(行5,列2-3)为Blue。

Green框分析(行6-9,列4-6)：
行6: G G K → 不完整。让我重新看。
(6,4)G (6,5)G
(7,4)G (7,6)G — (7,5)=K
(8,4)G (8,6)G — (8,5)=K
(9,4)G (9,5)G (9,6)G

上边(行6,列4-6): G G K → 缺失列6。不对，(6,6)=K但(7,6)=G。这不是一条完整缺失的边。

让我重新设计Green框使其有明确的缺失边：

修正Input:
```
K K K K K K K K K K
K B B B B K K K K K
K B K K B K K K K K
K B K K B K K K K K
K K K K K K K K K K
K K K K K G G G K K
K K K K K K K G K K
K K K B K K K G K K
K K K K K G G G K K
K K K K K K K K K K
```

Green框(行5-8,列5-7)：
行5: G G G（上边完整）
行6: K K G（右边列7有G）
行7: K K G（右边列7有G）
行8: G G G（下边完整）
左边(列5,行5-8): G K K G → 左边行6-7缺失 → 左边不完整，开口朝左。
右边(列7,行5-8): G G G G ✓

内部空腔=行6-7,列5-6（2行2列，但列5行6-7原本是框内部）。空腔深度（开口方向=左，深度=空腔列数=2）。开口高度=行6-7=2像素。
射线从开口处(行6-7,列4)向左发射，宽度2行，长度2列→填充(行6,列3-4)和(行7,列3-4)为Green。

诱饵：(4,7)处有一个孤立的Green像素G——但输入中(4,7)=K。(7,3)有一个孤立Blue像素B作为诱饵。

Output:
```
K K K K K K K K K K
K B B B B K K K K K
K B K K B K K K K K
K B K K B K K K K K
K K B B K K K K K K
K K K G G G G G K K
K K K G K K K G K K
K K K G B K K G K K
K K K K K G G G K K
K K K K K K K K K K
```

等等射线应从Blue框向下和Green框向左。

Blue框(行1-3,列1-4)下边缺失→射线向下，宽2(列2-3)，长2行→(行4,列2-3)为B B。
Green框(行5-8,列5-7)左边缺失→射线向左，宽2(行6-7)，长2列→(行6-7,列3-4)为G。

(7,3)的Blue诱饵和Green射线在(7,3)冲突吗？诱饵B在(7,3)，Green射线要覆盖(7,3)→"射线不覆盖其他已有的非Black像素"→(7,3)是B，非Black→Green不覆盖。

Output:
```
K K K K K K K K K K
K B B B B K K K K K
K B K K B K K K K K
K B K K B K K K K K
K K B B K K K K K K
K K K K K G G G K K
K K K G G K K G K K
K K K B G K K G K K
K K K K K G G G K K
K K K K K K K K K K
```

说明：Blue框(行1-3,列1-4)下边缺失→向下射线，宽=2(列2-3)，长=2行→(4,2)(4,3)填Blue。Green框(行5-8,列5-7)左边(列5)行6-7缺失→向左射线，宽=2(行6-7)，长=2列→(6,3)(6,4)(7,3)(7,4)应填Green，但(7,3)已有Blue诱饵→不覆盖→(6,3)(6,4)(7,4)填Green。诱饵Blue像素(7,3)保持不变。
</new_puzzle>

<new_puzzle>
<description>
网格中每个前景物体由两种颜色构成——大面积的"主体色"和仅1个像素的"标签色"（Red）。标签在物体bounding box中的相对位置（上半/下半/左半/右半）决定了该物体应执行的几何变换：标签在上半→整体上下翻转；标签在下半→保持不变；标签在左半→左右翻转；标签在右半→顺时针旋转90度。

Input structure:
- 背景为Black。
- 若干个前景物体，bounding box互不重叠。
- 每个物体在其bounding box内是一个非对称图案，由大面积的"主体色"和恰好1个Red像素（标签）构成。
- 每个物体的bounding box是正方形（宽=高），保证旋转后仍能放回原位。
- Red标签像素不在bounding box的正中央——它偏向某个半区。
- 所有物体的标签色统一为Red，主体色各不相同。

Rule:
- 对每个物体，找到其bounding box和内部唯一的Red像素位置。
- 判断Red像素在bounding box中的半区：
  - 行 < 中间行 → 上半区 → 将bounding box内全部内容上下翻转。
  - 行 > 中间行 → 下半区 → 保持原样不变。
  - 行 = 中间行且列 < 中间列 → 左半区 → 将bounding box内全部内容左右翻转。
  - 行 = 中间行且列 > 中间列 → 右半区 → 将bounding box内全部内容顺时针旋转90度。
- 变换包含Red像素本身，变换后放回原bounding box位置。
- 背景不变。

Key insight:
做题者看到不同物体发生了不同的几何变换（翻转、旋转、不变），自然会从物体的外部属性（颜色、大小、网格位置）寻找原因。物体内部那个微小的Red像素很容易被忽视或被当作"标记中心点"的装饰。真正的规则将Red像素的位置解读为变换类型的编码——这种位置→操作的映射不具有天然的直觉对应性（为什么"上半"对应"翻转"而非"上移"？），需要多个样例才能建立完整映射。
</description>

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K
K B B B K E E E K
K R K K K K K E K
K B K K K K R E K
K K K K K K K K K
```

物体1(Blue主体)在bounding box(行1-3,列1-3)，3x3正方形。
bbox内容：
```
B B B
R K K
B K K
```
Red在(2,1)→bbox内相对位置(行1,列0)(0-indexed within bbox)。bbox中间行=1，中间列=1。Red行=1=中间行，Red列=0<中间列=1→左半区→左右翻转。

左右翻转：
```
B B B      B B B
R K K  →   K K R
B K K      K K B
```

物体2(Grey/E主体)在bounding box(行1-3,列5-7)，3x3正方形。
bbox内容：
```
E E E
K K E
K R E
```
Red在(3,6)→bbox内相对位置(行2,列1)。中间行=1，中间列=1。Red行=2>中间行→下半区→保持不变。

Output:
```
K K K K K K K K K
K B B B K E E E K
K K K R K K K E K
K K K B K K R E K
K K K K K K K K K
```

说明：Blue物体Red标签在左半区→左右翻转→L形图案从左上角翻到右上角。Grey物体Red标签在下半区→不变。
</new_puzzle>

<new_puzzle>
<description>
网格中有一个由Grey像素构成的大型连通形状作为"模板"，模板外部散布着若干孤立的彩色单像素"探针"。每个探针的行和列分别在模板上截取一段，两段的交叉矩形区域内的模板像素被染成探针的颜色。

Input structure:
- 背景为Black。
- 一个由Grey像素组成的大型4-连通形状占据网格中部区域。
- 模板形状不一定是矩形——可能是L形、T形或其他不规则形状。
- 若干孤立的单像素彩色标记（非Grey、非Black）分布在模板外部。
- 每个标记所在的行穿过模板（该行上有Grey像素），且所在的列也穿过模板（该列上有Grey像素）。
- 不同标记的投影矩形不重叠（题目保证）。

Rule:
- 对每个彩色标记（位于row_m, col_m）：
  - 找到Grey模板在row_m行上的Grey像素列范围[c1, c2]（该行最左Grey到最右Grey的闭区间）。
  - 找到Grey模板在col_m列上的Grey像素行范围[r1, r2]（该列最上Grey到最下Grey的闭区间）。
  - 将模板中位于行[r1, r2]且列[c1, c2]矩形区域内的所有Grey像素着色为标记颜色。
- 不在矩形区域内的Grey像素保持Grey。
- 标记像素本身保留不变。
- 矩形区域内原本就是Black的像素（模板的"空洞"）保持Black。

Key insight:
做题者看到模板部分变色，首先会猜测"最近标记染色"或"泛洪填充"等简单规则。但着色区域是一个矩形子区，而非沿连通路径扩散的结果——且矩形的行范围和列范围分别来自两个不同的投影。做题者需要理解探针的行和列像"十字准星"一样在模板上取截面，两个截面的交叉区域才是着色范围。这种行列独立投影再求交集的操作在ARC题中罕见。
</description>

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K
K E E E E E K K
K E E E E E K K
B E E E E E K K
K E E E E E K K
K K K K E E K K
K K K K E E K K
K K K K K K K K
```

Grey模板：行1-4列1-5（矩形主体）+ 行5-6列4-5（下方突出部分）。

Blue探针在(3,0)。
- row_m=3: 行3上的Grey列范围 = 列1到列5 → [1,5]。
- col_m=0: 列0上无Grey像素。→ 无效？

col_m=0上没有Grey。修正探针位置使两个投影都有效。

修正Input:
```
K K K K K K K K
K E E E K K K K
K E E E K K K K
K E E E E E K K
K E E E E E K K
K K K K E E K K
K K K K E E K K
K K K K B K K K
```

Blue探针在(7,4)。
- row_m=7: 行7无Grey → 无效。

再修正：

Input:
```
K K B K K K K K
K E E E K K K K
K E E E K K K K
K E E E E E K K
K E E E E E K K
K K K K E E K K
K K K K E E K K
K K K K K K K K
```

Blue探针在(0,2)。
- row_m=0: 行0无Grey → 无效。

探针需要和模板在同行和同列都有Grey。让我将探针放在模板行范围内、列范围外的位置：

Input:
```
K K K K K K K K
K E E E K K K K
K E E E K K K K
K E E E E E K B
K E E E E E K K
K K K K E E K K
K K K K E E K K
K K K K K K K K
```

Blue在(3,7)。
- row_m=3: Grey列范围=[1,5]。✓
- col_m=7: 列7上无Grey。✗

需要探针所在列也穿过模板。列7上没有Grey。

Input:
```
K K K K K K K
K E E E K K K
K E E E K K K
K E E E E K K
K E E E E K K
K K E E E K K
K K E E E K K
K K K K K K K
K K K B K K K
```

Blue在(8,3)。
- row_m=8: 无Grey。

我需要探针在模板的行范围之内但不在模板像素上，且列也穿过模板。

让模板有一个凹陷，探针放在凹陷内：

Input:
```
K K K K K K K K
K E E K E E K K
K E E K E E K K
K E E E E E K K
K E E E E E K K
K E E E E E K K
K K K K K K K K
```

模板：行1-2列1-2，行1-2列4-5，行3-5列1-5。中间有凹陷：行1-2列3=K。

探针放在(1,3)——凹陷位置。但(1,3)=K且被Grey包围。

Blue探针在(1,3)。
- row_m=1: 行1Grey列={1,2,4,5}→列范围[1,5]。
- col_m=3: 列3Grey行={3,4,5}→行范围[3,5]。
- 着色区域：行[3,5]列[1,5]内的Grey像素。行3-5列1-5全是Grey→全变Blue。

Input:
```
K K K K K K K K
K E E B E E K K
K E E K E E K K
K E E E E E K K
K E E E E E K K
K E E E E E K K
K K K K K K K K
```

Output:
```
K K K K K K K K
K E E B E E K K
K E E K E E K K
K B B B B B K K
K B B B B B K K
K B B B B B K K
K K K K K K K K
```

说明：Blue探针在(1,3)，位于模板凹陷处（非Grey位置）。行1的Grey列范围=[1,5]。列3的Grey行范围=[3,5]。交叉矩形区域=行3-5列1-5内的Grey像素→全部着Blue。上方(行1-2)的Grey和凹陷位置不受影响。探针本身保留。
</new_puzzle>

<new_puzzle>
<description>
网格完全被若干不同颜色的4-连通区域（"领地"）无间隙覆盖。每个领地内部恰好有一个颜色不同的单像素"种子"。输出时每个领地执行一次颜色互换：领地的全部非种子像素变为种子的颜色，种子像素变为领地原来的颜色。

Input structure:
- 整个网格被若干不同颜色的4-连通区域完全平铺，没有背景色——每个像素都属于某个"领地"。
- 每个领地内部恰好包含一个与该领地颜色不同的单像素，即"种子"。
- 每个领地的种子颜色在全网格中是唯一的（不与其他领地的领地色或种子色重复）。
- 领地色和种子色的集合不重叠（没有一个领地的种子色恰好等于另一个领地的领地色）。

Rule:
- 对每个领地，识别其4-连通同色主体区域和内部唯一的异色种子像素。
- 将该领地所有主体像素（非种子）的颜色替换为种子的颜色。
- 将种子像素的颜色替换为领地原来的主体色。
- 即在每个领地内执行"主体色 ↔ 种子色"互换。

Key insight:
做题者首先注意到的是散布在大色块中的孤立异色像素，本能反应是"这些异色点会被消除或扩散"。真正的规则是角色反转——那个微小的异色种子才携带了领地的"目标颜色"，领地的大面积原色反而被压缩到种子一个像素上。"少数决定多数"的反转违背视觉直觉。做题者在看到第一个样例时可能猜到"颜色互换"，但需要验证种子色是否真的覆盖了整个领地、领地色是否真的缩回到种子位置——这种同时发生在多个领地上的全局互换需要仔细核对。
</description>

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
R R R B B B
R R R B M B
R G R B B B
A A A E E E
A O A E Y E
A A A E E E
```

领地分析：
- Red领地: {(0,0)(0,1)(0,2)(1,0)(1,1)(1,2)(2,0)(2,2)} → 种子(2,1)=Green
- Blue领地: {(0,3)(0,4)(0,5)(1,3)(1,5)(2,3)(2,4)(2,5)} → 种子(1,4)=Magenta
- Azure领地: {(3,0)(3,1)(3,2)(4,0)(4,2)} → 种子(4,1)=Orange
  修正：(4,0)=A, (4,1)=O, (4,2)=A → Azure主体={3,0-2 和 4,0 和 4,2}，种子=(4,1)=O ✓
  但(3,0)(3,1)(3,2)(4,0)(4,2)是否4-连通？(3,2)与(4,2)相邻✓，(4,0)与(3,0)相邻✓。都通过行3连通。✓
- Grey领地: {(3,3)(3,4)(3,5)(4,3)(4,5)} → 种子(4,4)=Yellow
  修正：(4,3)=E, (4,4)=Y, (4,5)=E → Grey主体={3,3-5 和 4,3 和 4,5}，种子=(4,4)=Y ✓
  但需要添加行5。(5,0-2)=A A A → 属于Azure？(5,0)与(4,0)=A相邻✓。Azure领地={3行0-2列,4行0列,4行2列,5行0-2列}，种子(4,1)=O。
  Grey领地={3行3-5列,4行3列,4行5列,5行3-5列}，种子(4,4)=Y。

完整领地：
- Red: 行0-2列0-2中除(2,1)外={8像素}，种子(2,1)=G
- Blue: 行0-2列3-5中除(1,4)外={8像素}，种子(1,4)=M
- Azure: 行3-5列0-2中除(4,1)外={8像素}，种子(4,1)=O
- Grey: 行3-5列3-5中除(4,4)外={8像素}，种子(4,4)=Y

互换后：
- Red→Green, 种子位→Red: 行0-2列0-2全变G, (2,1)变R
- Blue→Magenta, 种子位→Blue: 行0-2列3-5全变M, (1,4)变B
- Azure→Orange, 种子位→Azure: 行3-5列0-2全变O, (4,1)变A
- Grey→Yellow, 种子位→Grey: 行3-5列3-5全变Y, (4,4)变E

Output:
```
G G G M M M
G G G M B M
G R G M M M
O O O Y Y Y
O A O Y E Y
O O O Y Y Y
```

说明：每个3x3领地执行了"领地色↔种子色"互换。Red领地(8个像素)全变Green，种子位(2,1)从Green变Red。Blue领地变Magenta，种子位变Blue。Azure变Orange，种子位变Azure。Grey变Yellow，种子位变Grey。微小的种子像素"逆袭"成为领地的主色。
</new_puzzle>

<new_puzzle>
<description>
网格中有若干不同颜色的前景物体和一条Grey水平直线作为"镜面"。每个物体在镜面对侧生成反射像，但反射像不是完整副本——只保留物体中距镜面最远的那一行像素，且该行反射后出现在镜面对侧紧贴镜面的位置。反射像统一使用Maroon颜色，不保留原物体颜色。

Input structure:
- 背景为Black。
- 一条由Grey像素构成的完整水平线横贯网格。
- 镜面一侧（上方或下方）有若干个不同颜色的实心矩形前景物体。
- 镜面另一侧为空（全Black）。
- 物体不与镜面Grey线接触（至少间隔1行）。

Rule:
- 对每个物体，找到其距镜面最远的那一行（即物体最上面的行，如果物体在镜面上方）。
- 在镜面对侧紧贴镜面的位置（即镜面下方第一行），绘制该物体最远行的像素，但颜色统一替换为Maroon。
- 绘制时保持列位置不变（即最远行的哪些列有像素，反射行对应列就有Maroon像素）。
- 如果多个物体的反射在同一行的同一列重叠，仍为Maroon。
- 镜面本身和原物体不变。

Key insight:
做题者看到镜面线时会强烈预期标准的镜像反射——完整副本出现在对称位置。但输出只有一条Maroon线紧贴镜面，且不保留原色，完全打破了"完整镜像"的预期。做题者可能将这条Maroon线误认为镜面的装饰或边框。真正的规则提取了每个物体的"最远行"信息并将其压缩到镜面旁——这更像是"摘要"而非"反射"。用统一的Maroon色进一步隐藏了与原物体的颜色对应关系，增加了规则发现的难度。
</description>

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K
K B B K K K K K K
K B B K K G G G K
K B B K K G G G K
K K K K K G G G K
E E E E E E E E E
K K K K K K K K K
K K K K K K K K K
K K K K K K K K K
```

Blue物体(行1-3,列1-2)：距镜面(行5)最远的行=行1。行1的Blue像素在列1-2。
Green物体(行2-4,列5-7)：距镜面最远的行=行2。行2的Green像素在列5-7。

反射行=镜面下方第一行=行6。
行6列1-2画Maroon（来自Blue最远行），列5-7画Maroon（来自Green最远行）。

Output:
```
K K K K K K K K K
K B B K K K K K K
K B B K K G G G K
K B B K K G G G K
K K K K K G G G K
E E E E E E E E E
K W W K K W W W K
K K K K K K K K K
K K K K K K K K K
```

说明：镜面在行5。Blue(行1-3)最远行=行1(列1-2)→在行6列1-2画Maroon。Green(行2-4)最远行=行2(列5-7)→在行6列5-7画Maroon。所有反射像统一为Maroon色，紧贴镜面。原物体和镜面不变。
</new_puzzle>
