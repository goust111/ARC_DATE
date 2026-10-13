<new_puzzle>
- `description`：分隔线不是固定参考，而是根据两侧物体的存在关系发生局部偏移——有物体的一侧将分隔线推向对侧。

- `Input structure`：
  - 背景为 Black。
  - 一条完整的 Grey 水平线将网格分为上下两个区域。
  - 上下区域各有若干单色矩形块，每个矩形块的列范围互不重叠。

- `Rule`：
  - 对于分隔线上的每一列，检查该列上方和下方是否各有矩形块覆盖到该列。
  - 仅上方有块：该列的 Grey 像素下移 1 格（上方的块把分隔线往下"推"）。
  - 仅下方有块：该列的 Grey 像素上移 1 格（下方的块把分隔线往上"顶"）。
  - 两侧都有块或都没有块：该列的 Grey 像素不动。
  - 矩形块本身保持原位，仅分隔线发生形变。原分隔线位置的像素清为 Black，新位置绘制 Grey。

- `Key insight`：人类习惯将分隔线视为不可变的参考结构，关注两侧物体会如何被修改。但本题中分隔线才是被变换的唯一目标——它是输出中唯一改变的元素。这是"装饰元素即操作对象"的角色反转。物体仅充当控制信号，自身不改变。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K
K B B K K K K
K B B K K K K
E E E E E E E
K K K K R R K
K K K K R R K
K K K K K K K
```

Output:
```
K K K K K K K
K B B K K K K
K B B K K K K
E K K E K K E
K E E K R R K
K K K K R R K
K K K K K K K
```

说明：分隔线在行 3。列 0 两侧无块，Grey 不动。列 1-2 仅上方有 Blue 块（行 1-2），Grey 从行 3 下移到行 4。列 3 两侧无块，不动。列 4-5 仅下方有 Red 块（行 4-5），Grey 从行 3 上移到行 2——但行 2 列 4-5 是 Black 无块，所以实际检查：列 4 上方无块、下方有 Red → 上移到行 2。修正：列 4-5 下方有 Red（行 4-5 列 4-5），上方无块 → Grey 上移 1 格。但上移后行 2 列 4-5 画 Grey 会与行 2 的 Black 冲突——不会，那里本来就是 Black 背景。

修正 Output：
```
K K K K K K K
K B B K K K K
K B B K E E K
E K K E K K E
K E E K R R K
K K K K R R K
K K K K K K K
```

验证：列 0 两侧无块 → 行 3 不动。列 1 仅上方有 Blue → 行 3 的 Grey 下移到行 4，行 3 列 1 变 Black。列 2 同理下移。列 3 两侧无块 → 不动。列 4 仅下方有 Red → 行 3 的 Grey 上移到行 2，行 3 列 4 变 Black。列 5 同理上移。列 6 两侧无块 → 不动。最终分隔线变为折线形。
</new_puzzle>

<new_puzzle>
- `description`：网格中的矩形框架的边框上嵌有彩色标记。这些标记编码了对框架内部图案的行列置换：按标记颜色的排序结果重排内部的行和列。

- `Input structure`：
  - 背景为 Black。
  - 一个 Grey 像素构成的矩形框架（1 像素宽边框），框架内部填有多色图案。
  - 框架上边框（不含两个角）的某些 Grey 像素被替换为不同颜色，每列一种颜色（与内部列一一对应）。
  - 框架左边框（不含两个角）的某些 Grey 像素被替换为不同颜色，每行一种颜色（与内部行一一对应）。

- `Rule`：
  - 读取上边框从左到右的颜色序列（每种颜色对应内部一列），按 ARC 标准颜色序将它们排序（Blue < Red < Green < Yellow < Magenta < Orange < Azure < Maroon），排序后的顺序即为列的置换目标。
  - 读取左边框从上到下的颜色序列，同样排序，得到行的置换目标。
  - 对内部图案同时执行行列重排。
  - 输出中框架边框全部恢复为 Grey（彩色标记消失）。

- `Key insight`：人类看到边框上的彩色像素时，自然认为它们是标注或装饰。实际上这些颜色编码了一个排列置换。更关键的是，置换的定义方式是颜色排序——不是直接指示位置映射。做题者需要两步非直觉推理：发现边框标记控制了内部重排，然后发现重排规则是按颜色序排列。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
E G B R E
Y K O M E
R M K O E
B O M K E
E E E E E
```

Output:
```
E E E E E
B O M K E
R M K O E
Y K O M E
E E E E E
```

说明：框架 5×5，上边框内部列标记（列 1-3）：G, B, R。排序 B < G < R 对应原列 2, 1, 3，即新的列 1 取原列 2，新列 2 取原列 1，新列 3 取原列 3。左边框内部行标记（行 1-3）：Y, R, B。排序 B < R < Y 对应原行 3, 2, 1，即新行 1 取原行 3，新行 2 取原行 2，新行 3 取原行 1。

内部原始 3×3（行 1-3，列 1-3）：
行1: K O M
行2: M K O
行3: O M K

行重排（取原行 3, 2, 1）：
新行1: O M K
新行2: M K O
新行3: K O M

列重排（取原列 2, 1, 3）：
新行1: M O K
新行2: K M O
新行3: O K M

框架恢复全 Grey。

修正 Output：
```
E E E E E
M O K E E
K M O E E
O K M E E
E E E E E
```

再次检查：右边框和下边框本就是全 Grey，保持。内部重排后的 3×3 嵌入框架内（行 1-3，列 1-3）。列 4 和行 4 为框架右边和下边保持 Grey。验证通过。
</new_puzzle>

<new_puzzle>
- `description`：多条不同颜色的直线路径各从网格边缘向内延伸若干格后终止。每条路径沿原方向继续延伸到对侧边缘，但当穿过另一条路径时，穿过后的颜色变为被穿过路径的颜色。

- `Input structure`：
  - 背景为 Black。
  - 若干条单色直线段（水平或竖直），每条从网格的某个边缘出发向内延伸若干格后终止。
  - 不同路径颜色互不相同。
  - 路径在输入中不重叠，但它们的延长线可能相交。

- `Rule`：
  - 每条路径从终止端沿原方向延伸至网格对侧边缘。
  - 延伸时保持该路径的当前颜色。
  - 当延伸经过另一条已有路径（输入中原始像素或其延伸像素）所在的行或列时，从穿过点的下一个像素开始，颜色切换为被穿过路径的颜色。
  - 如果后续再穿过另一条路径，颜色再次切换为新穿过的路径颜色。
  - 交叉点本身保持被穿过路径（先存在于该位置的路径）的颜色。
  - 处理顺序：输入中路径越长的越先延伸；长度相同则按起始位置行号小的优先。

- `Key insight`：人类看到多条路径时，自然预期它们各自独立延伸或在交叉点停止。颜色在交叉后"传染"给穿过的路径，违反了路径独立性假设。做题者需要理解路径之间存在颜色传递关系——穿过不是终止而是变色的触发点。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K R K K K
K K K K K K
K K K K K K
K K K K K K
B B K K K K
K K K K K K
```

Output:
```
K K R K K K
K K R K K K
K K R K K K
K K R K K K
B B R B B B
K K B K K K
```

说明：Red 在 (0,2) 长度 1，Blue 在 (4,0)-(4,1) 长度 2。Blue 更长先延伸：Blue 向右延伸列 2-5，行 4。然后 Red 延伸：向下行 1-5 列 2。在 (4,2) 遇到 Blue 的延伸像素，交叉点保持 Blue（先存在），Red 继续但从 (5,2) 开始变 Blue 色。输出中 (4,2) 为 Blue 改为——Red 到达 (4,2) 时该位置已有 Blue，保持 Blue。(5,2) 画 Blue。

同时 Blue 延伸覆盖 (4,2)-(4,5)。但 (4,2) 在 Red 延伸时被 Red 经过——按规则交叉点保持先存在的颜色即 Blue。所以 (4,2) = Blue 的延伸结果。

但 Blue 延伸经过列 2 时，列 2 有 Red 的原始像素 (0,2)。Blue 在 (4,2) 处：(4,2) 是否算穿过 Red？Red 原始像素在 (0,2)，那是不同行的。Blue 沿行 4 水平延伸，在列 2 处没有遇到任何其他路径的像素（行 4 列 2 是 Black），所以 Blue 不穿过 Red，(4,2)-(4,5) 全为 Blue。

然后 Red 从 (0,2) 向下延伸：(1,2)(2,2)(3,2) 为 Red。(4,2) 已有 Blue → 交叉点保持 Blue，Red 穿过后变 Blue。(5,2) 为 Blue。

最终 Output：
```
K K R K K K
K K R K K K
K K R K K K
K K R K K K
B B B B B B
K K B K K K
```

(4,2) = Blue（先存在），行 4 全为 Blue 延伸加原始。验证通过。
</new_puzzle>

<new_puzzle>
- `description`：网格中有一个大型多色主体物体和若干外部的小型单色"模具"物体。每个模具的形状定义了从主体中挖去的区域——在主体的同色像素中找到唯一能容纳该模具形状的位置，将匹配像素清除为背景色。

- `Input structure`：
  - 背景为 Black。
  - 一个大型连通的多色物体占据网格中央区域，由多种颜色像素组成。
  - 物体外部散布 1-3 个小型单色"模具"物体，每个模具颜色在主体物体中也存在。

- `Rule`：
  - 对于每个模具，提取其形状（像素相对于包围盒左上角的偏移集合）。
  - 在主体物体中，找到与模具同色的所有像素构成的区域。
  - 在该同色区域中搜索唯一能完全容纳模具形状的位置（模具的每个偏移位置都落在同色像素上）。
  - 将该位置处匹配的像素全部清除为 Black。
  - 模具物体本身也清除为 Black。

- `Key insight`：人类看到小图案和大图案共存时，本能地预期"复制粘贴"——小图案应被印到大图案上。但本题方向相反：小图案是"减法模具"，定义了从大图案中移除的部分。此外，匹配不依赖空间邻近性而依赖颜色一致性和形状吻合性，增加了间接性。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K
K B B B K K
K R R B K K
K R R K K K
K K K K K K
K R R K K K
K R K K K K
```

Output:
```
K K K K K K
K B B B K K
K K K B K K
K K R K K K
K K K K K K
K K K K K K
K K K K K K
```

说明：主体物体在行 1-3 列 1-3，由 Blue{(1,1)(1,2)(1,3)(2,3)} 和 Red{(2,1)(2,2)(3,1)(3,2)} 构成。模具在行 5-6 列 1-2，颜色为 Red，形状为：
```
R R
R K
```
偏移集合：{(0,0),(0,1),(1,0)}。在主体 Red 区域 {(2,1)(2,2)(3,1)(3,2)} 中搜索：
- 起始 (2,1)：检查 (2,1)=R✓ (2,2)=R✓ (3,1)=R✓ → 全匹配。
- 起始 (2,2)：检查 (2,2)=R✓ (2,3)=B✗ → 不匹配。
- 起始 (3,1)：检查 (3,1)=R✓ (3,2)=R✓ (4,1)=K✗ → 不匹配。
唯一匹配位置 (2,1)。清除 (2,1)(2,2)(3,1) 为 Black。(3,2) 的 Red 保留。模具像素也清除。验证通过。
</new_puzzle>

<new_puzzle>
- `description`：每个前景物体向特定方向投射贯穿整行或整列的"影子射线"，方向由物体颜色编码。不同影子在同一像素重叠时产生输入中未出现的新颜色。

- `Input structure`：
  - 背景为 Black。
  - 若干单色前景像素或小型物体（1-2 个像素），散布在网格中。
  - 颜色可能包括 Red、Blue、Green、Yellow，每种颜色对应一个固定方向。

- `Rule`：
  - Red 物体的每个像素向右发射射线：从该像素的右邻开始，整行填充 Red 至右边缘。
  - Blue 物体的每个像素向下发射射线：从该像素的下邻开始，整列填充 Blue 至下边缘。
  - Green 物体的每个像素向左发射射线：从该像素的左邻开始，整行填充 Green 至左边缘。
  - Yellow 物体的每个像素向上发射射线：从该像素的上邻开始，整列填充 Yellow 至上边缘。
  - 如果恰好两条射线在同一像素重叠，该像素变为 Magenta。
  - 物体本身保持原色不变。

- `Key insight`：颜色编码方向与参考题 16（颜色编码有限位移）有表面相似性，但本题的关键突破在于射线是无限延伸的，且引入了重叠混色规则。输出中出现了 Magenta——一种输入中完全不存在的颜色。做题者会自然地在输入中搜索 Magenta 的来源，但它纯粹是两条射线交叉的产物。发现这一点需要从"输入映射输出"的逐像素思维转向"输出是多层叠加"的合成思维。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K
K K K K B K K
K K K K K K K
K R K K K K K
K K K K K K K
K K K K K K K
K K K K K K K
```

Output:
```
K K K K K K K
K K K K B K K
K K K K B K K
K R R R M R R
K K K K B K K
K K K K B K K
K K K K B K K
```

说明：Red 在 (3,1) 向右发射射线：(3,2)(3,3)(3,4)(3,5)(3,6) 填充 Red。Blue 在 (1,4) 向下发射射线：(2,4)(3,4)(4,4)(5,4)(6,4) 填充 Blue。交叉点 (3,4)：Red 射线和 Blue 射线重叠 → 变为 Magenta。其余射线像素保持各自颜色。物体本身 (3,1)=Red、(1,4)=Blue 不变。验证通过。
</new_puzzle>

<new_puzzle>
- `description`：若干带洞的环形岛屿纵向排列。每个岛屿的洞需要被填充，但填充色不取决于岛屿自身，而取决于该岛屿正上方最近的另一个岛屿的颜色。最顶部的岛屿洞保持为背景色。

- `Input structure`：
  - 背景为 Black。
  - 3-4 个不同颜色的单色环形物体（每个物体内部恰有一个被完全包围的矩形洞，即背景色区域）。
  - 物体之间不接触，上下分布在网格中。

- `Rule`：
  - 对每个岛屿，找到其内部洞（被该岛屿像素完全四面包围的 Black 区域）。
  - 从该岛屿的包围盒顶部向正上方搜索，找到最近的另一个岛屿（列范围与当前岛屿有重叠）。
  - 用该上方岛屿的颜色填充当前岛屿的洞。
  - 若上方没有其他岛屿，洞保持 Black。
  - 岛屿自身像素不变。

- `Key insight`：人类看到带洞的形状时，本能地从物体自身属性（颜色、大小）寻找填充规则。但洞的填色来自空间上的"邻居"——正上方最近的另一个物体。这是跨物体的远程信息传递，打破了"物体属性决定自身变换"的局部分析习惯。做题者需要从"自身特征"切换到"空间关系"这一非预期的关联维度。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K
K R R R R K K K
K R K K R K K K
K R R R R K K K
K K K K K K K K
K K B B B B K K
K K B K K B K K
K K B B B B K K
K K K K K K K K
```

Output:
```
K K K K K K K K
K R R R R K K K
K R K K R K K K
K R R R R K K K
K K K K K K K K
K K B B B B K K
K K B R R B K K
K K B B B B K K
K K K K K K K K
```

说明：Red 岛屿（行 1-3，列 1-4）内有洞（行 2，列 2-3）。上方无其他岛屿 → 洞保持 Black。Blue 岛屿（行 5-7，列 2-5）内有洞（行 6，列 3-4）。上方最近的岛屿是 Red（行 1-3，列 1-4，列范围与 Blue 洞的列 3-4 有重叠）→ 洞填充 Red。验证通过。
</new_puzzle>

<new_puzzle>
- `description`：网格最顶行是一个颜色调色板（位置编码数值），网格其余部分有成对的同色像素。每对同色像素之间画 L 形连线，但连线颜色不由端点颜色决定，而由两点的曼哈顿距离在调色板中查表得到。

- `Input structure`：
  - 背景为 Black。
  - 第 0 行为调色板行：列 0 为 Black，列 1 起每列一种不同颜色，依次编码数值 1、2、3……
  - 其余行中散布若干成对的同色像素，每种颜色恰好出现 2 次。

- `Rule`：
  - 对于每对同色像素 A 和 B，计算曼哈顿距离 D = |行差| + |列差|。
  - 查调色板行第 D 列的颜色 C。
  - 从 A 到 B 画 L 形路径：先沿水平方向移动到 B 的列，再沿竖直方向移动到 B 的行。
  - 路径上的中间像素（不含两个端点）全部填充为颜色 C。
  - 端点保持原色。调色板行不变。

- `Key insight`：人类看到成对像素后自然想用同色线连接。调色板行看起来像标题或装饰，容易被忽略。真正的难点有两层：首先连线颜色与端点颜色无关（由距离决定），其次"距离→颜色"的映射表就藏在网格顶部那行"装饰"里。这结合了角色反转（调色板看似背景实为核心数据）和非预期关联维度（颜色由距离而非属性决定）。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K B R G Y E M O
K K K K K K K K
K K K K K Y K K
K K K K K K K K
K K K K K K K K
K Y K K K K K K
K K K K K K K K
K K K K K K K K
```

Output:
```
K B R G Y E M O
K K K K K K K K
K K K K K Y K K
K K K K K O K K
K K K K K O K K
K Y O O O O K K
K K K K K K K K
K K K K K K K K
```

说明：Yellow 对在 (2,5) 和 (5,1)。曼哈顿距离 = |2-5| + |5-1| = 3 + 4 = 7。调色板列 7 = Orange。L 形路径从 (5,1) 先水平到 (5,5) 再竖直到 (2,5)。中间像素：(5,2)(5,3)(5,4)(5,5)(4,5)(3,5)——其中 (5,5) 是拐角属于中间像素，(2,5) 和 (5,1) 是端点保持 Yellow。所有中间像素填 Orange。验证通过。
</new_puzzle>
