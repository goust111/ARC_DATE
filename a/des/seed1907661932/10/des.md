<new_puzzle>
- `description`：网格中有若干彩色单像素"种子"嵌在一条 Grey 连通骨架上；所有种子同时沿骨架以相同速度进行 BFS 扩展，将 Grey 像素染为自身颜色；当两颗种子同步到达同一像素时该像素保持 Grey 作为边界。

- `Input structure`：
  - 背景为 Black。
  - 一条由 Grey 像素构成的单像素宽 4-连通骨架路径（可有分支，类似树状结构）。
  - 2-4 个不同颜色的单像素种子，每颗种子恰好位于骨架上某个位置（替代了该处的 Grey）。种子颜色互不相同，且不为 Black 或 Grey。

- `Rule`：
  - 从每颗种子同时开始，沿骨架的 4-连通邻居进行 BFS 扩展，每步前进一个像素。
  - 某个 Grey 像素被第一个到达它的种子染色。若两颗种子同步到达（沿骨架的路径距离相等），该像素保持 Grey 作为边界。
  - 扩展只沿骨架进行，不穿过 Black 背景，也不穿过已被其他种子占据的像素。
  - 种子自身的位置保持其原色不变。
  - Black 背景不参与任何变化。

- `Key insight`：
  - 做题者很可能尝试用欧几里得距离判断每个像素"最近的种子"，但实际距离是沿骨架路径的测地距离。在弯曲或分支的骨架上，空间上看起来更近的种子可能因路径绕行而实际更远。等距边界保持 Grey 的规则打破了"所有 Grey 都会消失"的预期。核心难点在于从"空间邻近"思维切换到"拓扑路径距离"思维。

验证样例（K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K
R E E E E E B
K K K K K K K
```

Output:
```
K K K K K K K
R R R E B B B
K K K K K K K
```

说明：骨架为第 1 行从列 0 到列 6 的水平直线（共 7 格）。种子 R 在 (1,0)，种子 B 在 (1,6)。R 向右扩展，B 向左扩展。R 到 (1,1)=1步, (1,2)=2步, (1,3)=3步。B 到 (1,5)=1步, (1,4)=2步, (1,3)=3步。(1,1)(1,2) 只有 R 先到，染 R；(1,5)(1,4) 只有 B 先到，染 B；(1,3) 两者同步到达（均 3 步），保持 Grey。
</new_puzzle>

<new_puzzle>
- `description`：网格中散布着若干对形状完全相同的同色物体，每对之间有一个颜色独特的单像素"开关"；开关在间隙中偏向哪个物体，就保留那个物体，删除另一个。

- `Input structure`：
  - 背景为 Black。
  - 网格中有若干颜色各异的物体对。每种前景色恰好出现两个形状完全相同的 4-连通物体。
  - 每对同色物体沿水平或垂直方向排列（左右或上下分布），两者之间有 Black 间隙。
  - 每对物体的间隙中恰好有一个颜色独特的单像素（"开关"），该开关颜色在整个网格中只出现一次。

- `Rule`：
  - 对每对同色物体，找到它们之间的唯一开关单像素。
  - 若物体对水平排列：如果开关的列坐标距左边物体的包围盒右边缘更近，保留左边物体，删除右边物体；若距右边物体的包围盒左边缘更近，保留右边物体，删除左边物体。
  - 若物体对垂直排列：同理按行坐标与上下物体的距离判断。
  - 开关偏向哪个物体就保留那个物体。
  - 被删除的物体和开关像素都变为 Black。
  - 被保留的物体保持原位原色不变。

- `Key insight`：
  - 做题者会注意到成对的同色物体并试图寻找形状差异来判定保留哪个，但两个物体形状完全相同。接着可能按位置规律（如总是保留左边的）来猜测，但不同物体对保留的方向不一致。真正的线索是物体之间那个容易被当作噪声的单个异色像素——它的偏向位置决定了选择。难点在于发现这个不起眼的点是核心决策依据而非噪声。

验证样例（K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K
K R R K K K K R R
K K R K K E K K R
K K K K K K K K K
K K K K K K K K K
K B K K K K K B K
K B B K O K K B B
K K K K K K K K K
```

Output:
```
K K K K K K K K K
K K K K K K K R R
K K K K K K K K R
K K K K K K K K K
K K K K K K K K K
K B K K K K K K K
K B B K K K K K K
K K K K K K K K K
```

说明：
- Red 对水平排列：左 Red 在 (1,1)(1,2)(2,2)，右 Red 在 (1,7)(1,8)(2,8)，形状均为 L 形。开关 Grey(E) 在 (2,5)。左 Red 右边缘列 2 到开关列 5 距离 3；右 Red 左边缘列 7 到开关列 5 距离 2。开关偏向右 Red，保留右 Red，删除左 Red 和开关。
- Blue 对水平排列：左 Blue 在 (5,1)(6,1)(6,2)，右 Blue 在 (5,7)(6,7)(6,8)，形状均为 L 形。开关 Orange(O) 在 (6,4)。左 Blue 右边缘列 2 到开关列 4 距离 2；右 Blue 左边缘列 7 到开关列 4 距离 3。开关偏向左 Blue，保留左 Blue，删除右 Blue 和开关。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干封闭矩形框和一条横贯的直线；直线将每个框的内部分为两个区域，保留填充像素更多的区域的内容，清除填充像素更少的区域的内容。

- `Input structure`：
  - 背景为 Black。
  - 若干个由 Grey 像素构成的一像素宽空心矩形框，框之间不重叠。
  - 每个框内部有若干该框独有颜色的像素（部分填充的图案）。
  - 一条由 Azure 像素构成的一像素宽水平或垂直直线贯穿整个网格宽度或高度，穿过所有矩形框。该直线绘制在框和内容之上。

- `Rule`：
  - Grey 矩形框和 Azure 直线保持不变。
  - 对每个矩形框，Azure 直线将框内部（不含框边界和直线像素）分为两个子区域。
  - 统计每个子区域中非 Black 彩色像素的数量。
  - 保留像素数较多的子区域的内容不变。
  - 将像素数较少的子区域中的所有彩色像素清除为 Black。
  - 两侧相等时保留上方或左方区域。

- `Key insight`：
  - 做题者会以为直线是镜像轴（内容被对称复制）或统一裁剪线（总是保留同一侧），但不同框被保留的方向可能不同——有的保留上半，有的保留下半。做题者需要多个样例才能排除固定方向假设，发现是逐框比较数量后多数决定。难点在于隐含的条件分支：每个框独立决策，保留方向由各自内容分布决定。

验证样例（K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
E E E E K E E E E
E B B E K E K R E
E K B E K E K K E
A A A A A A A A A
E K K E K E R R E
E E E E K E E E E
```

Output:
```
E E E E K E E E E
E B B E K E K K E
E K B E K E K K E
A A A A A A A A A
E K K E K E R R E
E E E E K E E E E
```

说明：Azure 水平直线在第 3 行。左框 Grey 边界为行 0、5 和列 0、3，内部区域分为上半（行 1-2，列 1-2）和下半（行 4，列 1-2）。上半 Blue 像素：(1,1)(1,2)(2,2) = 3 个。下半 Blue 像素：0 个。上半更多→保留上半（无变化）。右框 Grey 边界为行 0、5 和列 5、8，内部区域分为上半（行 1-2，列 6-7）和下半（行 4，列 6-7）。上半 Red 像素：(1,7) = 1 个。下半 Red 像素：(4,6)(4,7) = 2 个。下半更多→保留下半，清除上半的 (1,7) R→K。两个框保留了不同的侧面，体现逐框独立决策的核心规则。
</new_puzzle>

<new_puzzle>
- `description`：网格由分隔线划分为若干面板，每个面板内有一个 Grey 图案和一个角落颜色标签；将所有面板的图案按扫描顺序依次叠加到一个面板大小的画布上——Grey 位置涂该面板的标签色并覆盖已有内容，Black 位置不改变画布内容。

- `Input structure`：
  - 网格被 Maroon 一像素宽分隔线分割为若干大小相等的矩形面板（如 1xN 或 2x2 排列）。
  - 每个面板内部由 Black 背景和若干 Grey 像素构成的图案组成。
  - 每个面板的右下角有一个非 Black、非 Grey、非 Maroon 的单像素作为该面板的颜色标签。
  - 不同面板标签颜色各不相同。

- `Rule`：
  - 输出大小等于单个面板大小。
  - 初始化输出画布为全 Black。
  - 按面板在网格中从左到右、从上到下的顺序依次处理每个面板：
    - 对于面板中每个 Grey 像素的位置，将画布对应位置设为该面板的标签颜色（覆盖已有内容）。
    - 对于面板中 Black 像素的位置，不修改画布（保留之前的内容）。
    - 标签像素的位置不参与图案叠加（视为 Black）。
  - 最终画布即为输出。

- `Key insight`：
  - 做题者最初会把多面板布局理解为"多个独立样例"（ARC 常见模式），试图在每个面板中找出独立的变换规则。但这些面板不是独立样例——它们是按顺序叠加的半透明图层。Grey 是"不透明"（覆盖下层），Black 是"透明"（让下层内容透过来）。角落的单色标签点容易被当作噪声，但它决定了每个图层的颜色身份。核心难点是打破"面板=独立样例"的心理预设，转而理解为图层合成。

验证样例（K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
E E K W K E K
K E K W E K K
K K R W K K G
```

Output:
```
R G K
G R K
K K K
```

说明：Maroon 分隔线在列 3，将 3x7 网格分为左面板 3x3（列 0-2）和右面板 3x3（列 4-6）。
- 左面板内容（列 0-2）：E E K / K E K / K K R。Grey 在 (0,0)(0,1)(1,1)。标签 Red 在 (2,2)。
- 右面板内容（列 4-6）：K E K / E K K / K K G。Grey 在 (0,1)(1,0)。标签 Green 在 (2,2)。
- 画布初始全 K（Black）。
- 第 1 步（左面板，标签 Red）：Grey 位置 (0,0)(0,1)(1,1) → 画布 (0,0)=R, (0,1)=R, (1,1)=R。画布：R R K / K R K / K K K。
- 第 2 步（右面板，标签 Green）：Grey 位置 (0,1)(1,0) → 画布 (0,1)=G（覆盖 R）, (1,0)=G。画布：R G K / G R K / K K K。
- (0,0) 仅左面板有 Grey → Red。(0,1) 两面板都有 Grey，右面板后处理 → Green 覆盖 Red。(1,0) 仅右面板 → Green。(1,1) 仅左面板 → Red。其余位置无 Grey → 保持 Black。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干不同颜色的矩形块，每个矩形块恰好缺少一个像素（凹口在包围盒的某条边上）；凹口所在边的方向编码该块的推动方向，将块沿该方向滑到网格边缘或碰到障碍后停下，并填补凹口使之成为完整矩形。

- `Input structure`：
  - 背景为 Black。
  - 若干不同颜色的近矩形块散布在网格中，互不重叠。
  - 每个块是一个完整矩形减去一个像素：缺失像素（凹口）位于包围盒四条边之一的非角落位置，该位置为 Black（与背景一致）。
  - 各块颜色互不相同。

- `Rule`：
  - 确定每个块凹口所在的边：凹口在上边→向上推；凹口在下边→向下推；凹口在左边→向左推；凹口在右边→向右推。
  - 按推动方向到目标边界的距离从近到远排序处理（最近的块先停下，后续块可以碰到已停下的块）。
  - 将块整体沿推动方向滑动，直到前沿碰到网格边界或已停下的块。
  - 滑动完成后，填补凹口像素为该块颜色，使之成为完整矩形。
  - 原始位置清为 Black。

- `Key insight`：
  - 做题者首先注意到矩形块各有一个小缺口，最直接的猜测是"填补缺口"就完事。但输出中块的位置发生了改变，仅填补不能解释。接着做题者可能按颜色、大小或位置来猜测运动方向，但方向其实由凹口朝向决定——将"缺损"理解为"方向箭头"是反直觉的。额外难点是块之间可能碰撞堆叠，处理顺序影响最终布局。

验证样例（K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K
K K K K K K K
K R R R K K K
K R R R K K K
K R K R K K K
K K K K K K K
K K K K K K K
```

Output:
```
K K K K K K K
K K K K K K K
K K K K K K K
K K K K K K K
K R R R K K K
K R R R K K K
K R R R K K K
```

说明：Red 块包围盒 (2,1)-(4,3) 为 3x3，缺口在 (4,2)（下边中间）→向下推。块从行 2-4 向下滑动，直到前沿碰到网格底边行 6：新位置行 4-6 列 1-3。填补缺口使之成为完整 3x3 Red 矩形。原位置行 2-3 清为 Black。
</new_puzzle>

<new_puzzle>
- `description`：网格中有一个大的多色参考图案和若干被 Grey 边框围起来的小测试块；每个测试块是参考图案某个子区域的复制品但其中恰好有一个像素颜色被篡改；将被篡改的像素修正为参考图案中对应位置的正确颜色。

- `Input structure`：
  - 背景为 Black。
  - 网格一角有一个较大的多色参考图案（至少 3x3），由多种非 Black 非 Grey 颜色构成。
  - 网格其他位置有 1-3 个小矩形区域，每个被一像素宽 Grey 边框围起来。
  - 每个 Grey 框内部是参考图案某个等大子区域的复制品，但恰好有一个像素的颜色与参考中对应位置不同。
  - 通过内容匹配可唯一确定每个测试块对应参考图案中的哪个子区域（最大匹配只差一个像素的位置）。

- `Rule`：
  - 参考图案和 Grey 边框保持不变。
  - 对每个测试框内的子区域，在参考图案中找到与之仅差一个像素的对应子区域。
  - 将测试框内该差异像素的颜色修正为参考图案中对应位置的正确颜色。
  - 其余像素不变。

- `Key insight`：
  - 做题者可能以为每个小框是独立的变换样例，或以为大图案是某种"调色板"或"图例"。真正的关系是"原版-副本"的子区域对齐——测试框是从大图案中抄出来的片段，内藏一个抄写错误。难点在于发现这种跨区域的像素级比对关系，尤其当变化只有一个像素时不容易直接看出哪个像素出了问题。

验证样例（K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
R B G K K K K K K
B A R K K K K K K
G R B K K K K K K
K K K K K K K K K
K K K K E E E E E
K K K K E R B G E
K K K K E B B R E
K K K K E G R B E
K K K K E E E E E
```

Output:
```
R B G K K K K K K
B A R K K K K K K
G R B K K K K K K
K K K K K K K K K
K K K K E E E E E
K K K K E R B G E
K K K K E B A R E
K K K K E G R B E
K K K K E E E E E
```

说明：参考图案 3x3 在 (0,0)-(2,2)：R B G / B A R / G R B。测试框在 (4,4)-(8,8)，Grey 边框围起内部 3x3 在 (5,5)-(7,7)：R B G / B B R / G R B。与参考逐像素对比：(6,6) 处测试为 B，参考 (1,1) 为 A(Azure)，其他位置全部一致。将 (6,6) 从 B 修正为 A。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干不同颜色的封闭矩形框，每个框内部散落着若干 Grey 标记像素；统计每个框内 Grey 标记的数量 N，然后将该框的边界向内膨胀 N 层，框内 Grey 标记清除。

- `Input structure`：
  - 背景为 Black。
  - 若干不同颜色的一像素宽空心矩形框散布在网格中，每种颜色只出现一个框，框之间不重叠不嵌套。
  - 每个框内部有 0-3 个 Grey 单像素散落在不同位置，其余内部像素为 Black。

- `Rule`：
  - 对每个框：
    1. 统计内部 Grey 像素数量 N，然后将所有 Grey 清除为 Black。
    2. 从框的一像素宽边界开始，向内部膨胀 N 层。第 1 层：将与框边界 4-相邻的内部 Black 像素填充为框颜色。第 2 层：将与第 1 层新增像素 4-相邻的剩余内部 Black 像素填充为框颜色。以此类推。
    3. N=0 时边界保持原样不膨胀。若膨胀层数超过内部可用空间则完全填满。
  - Black 背景不变。

- `Key insight`：
  - 做题者首先关注不同框的形状、大小和颜色差异，试图从几何属性找变换规则。框内散落的灰色小点很不起眼，容易被当作噪声。但灰色点的数量恰恰是控制膨胀程度的"指令参数"——它们是嵌入画面中的元数据。难点在于：(1) 灰色标记不是图案而是操作参数；(2) 膨胀方向是向内收缩空间而非向外扩展。

验证样例（K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K K K K
K R R R R R K K K K K K
K R K K K R K K K K K K
K R K E K R K B B B B K
K R K K K R K B K E B K
K R R R R R K B K E B K
K K K K K K K B B B B K
K K K K K K K K K K K K
```

Output:
```
K K K K K K K K K K K K
K R R R R R K K K K K K
K R R K K R K K K K K K
K R K K K R K B B B B K
K R R K K R K B B B B K
K R R R R R K B B B B K
K K K K K K K B B B B K
K K K K K K K K K K K K
```

说明：
- Red 框边界 (1,1)-(5,5)，内部 (2,2)-(4,4) 共 3x3。有 1 个 Grey 在 (3,3)，N=1。清除 Grey 后膨胀 1 层：与边界 4-相邻的内部像素 (2,2)(2,3)(2,4)(3,2)(3,4)(4,2)(4,3)(4,4) 全填 Red。中心 (3,3) 不与边界相邻，保持 Black。
- Blue 框边界 (3,7)-(6,10)，内部 (4,8)-(5,9) 共 2x2。有 2 个 Grey 在 (4,9)(5,9)，N=2。清除 Grey 后膨胀 2 层：第 1 层填内部与边界相邻的全部像素——2x2 空间膨胀 1 层已经全部填满。内部完全变为 Blue。
</new_puzzle>
