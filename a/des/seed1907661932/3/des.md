<new_puzzle>
- `description`：网格中散布着若干彩色像素，同时有若干Grey像素对构成"传送门"。每个彩色像素沿传送门对进行距离对称的跨越传送——从一端外侧穿越到另一端外侧的对称位置。

- `Input structure`：
  - 背景色为Black。
  - 若干个单独的彩色像素（非Grey、非Black）散布在网格中。
  - 若干对Grey像素，每对恰好在同一行上（水平传送门）或同一列上（垂直传送门），每对之间没有其他Grey像素。
  - 每个彩色像素恰好与一对Grey传送门共处同一行或同一列，并且位于传送门对的外侧（不在两个Grey像素之间）。

- `Rule`：
  - 对每个彩色像素P，找到与它共行或共列的Grey传送门对(G1, G2)，其中G1是距P较近的那个Grey像素。
  - 计算P到G1的距离d（像素间隔数）。
  - 在G2的另一侧（远离G1的方向），距离G2同样为d的位置放置一个与P同色的新像素。
  - 擦除原始位置的P。
  - Grey传送门像素保持不变。

- `Key insight`：做题者首先注意到彩色像素和Grey像素的存在，容易猜测Grey是障碍物、边界标记或路径端点。实际上每对Grey像素构成一个"传送门"——彩色像素以距离对称的方式穿过一对Grey点出现在另一侧。核心难点在于发现Grey像素是成对工作的传送装置，且传送保持距离不变。人类倾向于独立解读每个Grey像素的功能，而非将两个Grey像素视为一个协同的传送对。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K
K R K E K K E K K
K K K K K K K K K
K K K K K K K K K
K K K K K K K K K
K K E K K K K K K
K K K K K K K K K
K K E K K K K K K
K K K K K K K K K
K K B K K K K K K
```

Output:
```
K K K K K K K K K
K K K E K K E K R
K K K K K K K K K
K K B K K K K K K
K K K K K K K K K
K K E K K K K K K
K K K K K K K K K
K K E K K K K K K
K K K K K K K K K
K K K K K K K K K
```

简要说明此样例如何体现规则：Red像素在(1,1)，同行传送门对为(1,3)和(1,6)。Red在(1,3)左侧，距离d=3-1=2。在(1,6)右侧距离2处放Red→(1,8)。擦除(1,1)。Blue像素在(9,2)，同列传送门对为(5,2)和(7,2)。Blue在(7,2)下方，距离d=9-7=2。在(5,2)上方距离2处放Blue→(3,2)。擦除(9,2)。Grey像素全部保持不变。
</new_puzzle>

<new_puzzle>
- `description`：网格被若干条完整的水平彩色分隔线划分为条带，分隔线的颜色不是装饰而是排序指令——每条分隔线的颜色指定其下方条带应放置哪个颜色的色块，从而对色块进行重排。

- `Input structure`：
  - 背景色为Black。
  - 网格被若干条完整的水平单像素彩色线（占满整行）分隔为多个水平条带。
  - 每条分隔线颜色各不相同，且恰好与某个条带内的色块颜色匹配。
  - 每个条带（相邻两条分隔线之间的区域）内恰好有一个单色实心矩形色块（位于背景中），色块的高度等于条带高度。
  - 分隔线颜色集合与色块颜色集合完全一致（构成一一对应）。

- `Rule`：
  - 读取从上到下每条分隔线的颜色序列，记为C1, C2, C3, ...。
  - 第k条分隔线颜色为Ck，表示颜色Ck的色块应被移到第k条分隔线正下方的条带中。
  - 色块移动后保持原形状和大小，水平居中放置在目标条带中。
  - 分隔线本身保持不变。

- `Key insight`：做题者第一反应会将分隔线当作纯粹的结构性边界而忽略其颜色信息——毕竟分隔线看起来就是"画框"。实际上分隔线不是装饰，它们的颜色序列构成了色块重排的指令表。核心难点在于意识到"边界元素本身携带语义信息"——这是角色反转：看似被动的分割线实际是主动的控制指令。

验证样例（颜色首字母：K=Black, R=Red, B=Blue, G=Green）：

Input:
```
R R R R R R R R
K K B B K K K K
K K B B K K K K
G G G G G G G G
K K K K K R R K
K K K K K R R K
B B B B B B B B
K G G K K K K K
K G G K K K K K
```

Output:
```
R R R R R R R R
K K K R R K K K
K K K R R K K K
G G G G G G G G
K K K G G K K K
K K K G G K K K
B B B B B B B B
K K K B B K K K
K K K B B K K K
```

简要说明此样例如何体现规则：三条分隔线从上到下颜色依次为Red(行0)、Green(行3)、Blue(行6)。规则：Red线下方放Red色块→原Red色块(2x2)从条带2(行4-5列5-6)移到条带1(行1-2)并居中至列3-4；Green线下方放Green色块→原Green色块(2x2)从条带3(行7-8列1-2)移到条带2(行4-5)并居中至列3-4；Blue线下方放Blue色块→原Blue色块(2x2)从条带1(行1-2列2-3)移到条带3(行7-8)并居中至列3-4。所有色块在8列宽的条带中居中放置于列3-4。
</new_puzzle>

<new_puzzle>
- `description`：每个封闭矩形框架内有一个偏离中心的种子像素，种子到四面内壁的距离中唯一最短的方向决定了种子向反方向射出直线——种子"逃离"它最近的墙壁。

- `Input structure`：
  - 背景色为Black。
  - 若干个由单像素宽的彩色边框围成的矩形封闭区域，彼此不重叠。
  - 每个区域内部初始为背景色，且恰好包含一个单像素的种子（颜色与边框不同，非背景色）。
  - 每个种子的位置使得它到四面内壁的距离存在唯一最小值（不存在并列最短）。

- `Rule`：
  - 对每个封闭矩形区域，计算种子像素到四面内壁的距离（上、下、左、右各一个值，即种子到该方向最近的边框内侧的像素间隔数）。
  - 找到距离唯一最短的方向——种子最靠近的那面墙。
  - 从种子像素出发，沿该最短方向的反方向（远离最近墙壁）画一条直线，用种子的颜色填充每个像素，直到碰到对面的内壁（不覆盖边框像素）。
  - 种子像素本身保留在原位，边框不变。

- `Key insight`：做题者最先猜测的是种子向最近的墙发射（直觉上"撞向最近障碍"），或者向四个方向都发射。实际规则是"远离最近墙"——种子逃离它最靠近的边界。这是一个反直觉的方向选择，需要做题者跨越"靠近→远离"的认知逆转。同时，信息（最近墙的位置）和动作（射向反方向）之间存在间接关系，做题者需要理解距离不对称性是方向选择的编码方式。

验证样例（颜色首字母：K=Black, E=Grey, R=Red, B=Blue）：

Input:
```
K K K K K K K K K
K E E E E E E E K
K E K K K K K E K
K E K K K K K E K
K E K K K K K E K
K E K R K K K E K
K E K K K K K E K
K E K K K K K E K
K E E E E E E E K
K K K K K K K K K
```

Output:
```
K K K K K K K K K
K E E E E E E E K
K E K K K K K E K
K E K K K K K E K
K E K K K K K E K
K E K R R R R E K
K E K K K K K E K
K E K K K K K E K
K E E E E E E E K
K K K K K K K K K
```

简要说明此样例如何体现规则：Grey边框围成矩形（行1和行8为水平边，列1和列7为垂直边），内部可用区域为行2-7、列2-6。Red种子在(5,3)。距离计算：上=5-2=3，下=7-5=2，左=3-2=1（唯一最短），右=6-3=3。最近墙在左方（距离1），反方向为右。从(5,3)出发向右画Red直线至内壁列6：(5,4)=R, (5,5)=R, (5,6)=R。种子(5,3)保留R。输出中行5变为 K E K R R R R E K。
</new_puzzle>

<new_puzzle>
- `description`：网格中有多个单色实心矩形，每个矩形被填充为棋盘格图案，但棋盘格的相位由全局网格坐标(r+c)的奇偶性统一决定，而非每个矩形的局部左上角坐标。

- `Input structure`：
  - 背景为Black的网格。
  - 若干个不同大小的实心矩形，每个矩形由单一前景色均匀填充（不同矩形颜色可不同）。
  - 矩形之间不重叠、不相邻。

- `Rule`：
  - 对每个矩形的每个像素位置(r,c)：若(r+c)为偶数，保持矩形原色不变；若(r+c)为奇数，替换为Azure。
  - 非矩形区域的Black背景保持不变。
  - 所有矩形共享同一个全局相位判据，不存在逐矩形的局部相位偏移。

- `Key insight`：做题者最初会以为每个矩形独立生成棋盘格，且每个矩形的左上角总是保持原色。但当两个矩形的左上角全局坐标(r+c)奇偶性不同时，一个矩形左上角保持原色而另一个变为Azure——这打破了"局部坐标决定相位"的假设。做题者需要从"每个矩形独立变换"的局部思维切换到"所有矩形共享同一个全局棋盘格"的全局视角。两个样例就够展示这种不一致，但需要第三个样例来排除其他假设。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K
K R R R K K K
K R R R K K K
K K K K K K K
K K K G G K K
K K K G G K K
K K K K K K K
```

Output:
```
K K K K K K K
K R A R K K K
K A R A K K K
K K K K K K K
K K K A G K K
K K K G A K K
K K K K K K K
```

简要说明此样例如何体现规则：Red矩形行1-2列1-3，按全局(r+c)：(1,1)=2偶→R, (1,2)=3奇→A, (1,3)=4偶→R, (2,1)=3奇→A, (2,2)=4偶→R, (2,3)=5奇→A。Green矩形行4-5列3-4：(4,3)=7奇→A, (4,4)=8偶→G, (5,3)=8偶→G, (5,4)=9奇→A。关键观察：Green矩形左上角(4,3)的(r+c)=7为奇数，因此左上角变为Azure而非保持Green——这与Red矩形左上角(1,1)保持原色的行为不同，证明相位是全局的而非局部的。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干对形状互补的小图案——一个图案的有色像素恰好填补另一个图案的空白像素。配对后，将"钥匙"图案的颜色嵌入"锁孔"图案的空白位置形成双色完整方形，钥匙原位清空。

- `Input structure`：
  - 背景为Black。
  - 网格中有偶数个占据相同大小方形区域（如3x3）的小图案，彼此不相邻。
  - 每个图案在方形区域中由单一前景色填充部分像素，其余位置为Black。
  - 这些图案两两配对形成互补关系：一对中，图案A的有色像素位置集合与图案B的有色像素位置集合合在一起恰好覆盖整个方形区域的所有位置，无重叠无遗漏。
  - 配对的两个图案颜色不同。每个图案恰好与一个图案互补配对。
  - 每对中像素数较多的为"锁孔"，较少的为"钥匙"。

- `Rule`：
  - 找到每对互补的图案（钥匙与锁孔）。
  - 在锁孔位置：保留其原有色像素不变，将其Black像素（凹陷部分）填充为配对钥匙的颜色，形成双色完整方形。
  - 钥匙的原始位置完全清空为Black。

- `Key insight`：做题者习惯按颜色相似性、空间邻近性或形状相似性来寻找图案之间的关联。但配对的真正逻辑是几何互补——一个图案的有色像素恰好填入另一个图案的Black空缺。做题者需要将Black背景视为有意义的形状信息而非"空白"，完成从"前景匹配"到"正负空间互补"的认知跳转。当多个图案同时存在时，正确配对需要逐个检验互补性。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K
K B B K K K K G K
K B K K K K G G K
K K K K K G G G K
K K K K K K K K K
K K R K K Y K Y K
K R R K K K K Y K
K K R K K Y K Y K
K K K K K K K K K
```

Output:
```
K K K K K K K K K
K K K K K B B G K
K K K K K B G G K
K K K K K G G G K
K K K K K K K K K
K K K K K Y R Y K
K K K K K R R Y K
K K K K K Y R Y K
K K K K K K K K K
```

简要说明此样例如何体现规则：所有图案均为3x3。Blue图案(行1-3,列1-3)有色位置(局部坐标)：(0,0)B (0,1)B (1,0)B，共3个像素，Black位置6个。Green图案(行1-3,列5-7)有色位置(局部)：(0,2)G (1,1)G (1,2)G (2,0)G (2,1)G (2,2)G，共6个像素，Black位置：(0,0)(0,1)(1,0)共3个。Blue有色∪Green有色 = {(0,0)(0,1)(1,0)} ∪ {(0,2)(1,1)(1,2)(2,0)(2,1)(2,2)} = 全部9个位置，互补成立。Green(6个)为锁孔，Blue(3个)为钥匙。锁孔的Black空缺{(0,0)(0,1)(1,0)}填Blue色→行1列5-7变为[B,B,G]，行2列5-7变为[B,G,G]，行3列5-7变为[G,G,G]。钥匙位置(行1-3,列1-3)清空为全Black。Red图案(行5-7,列1-3)有色(局部)：(0,1)R (1,0)R (1,1)R (2,1)R，共4个。Yellow图案(行5-7,列5-7)有色(局部)：(0,0)Y (0,2)Y (1,2)Y (2,0)Y (2,2)Y，共5个。Red∪Yellow = {(0,1)(1,0)(1,1)(2,1)} ∪ {(0,0)(0,2)(1,2)(2,0)(2,2)} = 全部9个位置，互补成立。Yellow(5个)为锁孔，Red(4个)为钥匙。锁孔Black空缺{(0,1)(1,0)(1,1)(2,1)}填Red色。钥匙位置清空。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干个由不同颜色边框围成的矩形房间，房间边框上有单像素缺口（门）。根据每个房间门数量的奇偶性决定是否填充房间内部——奇数门则用边框颜色填充内部，偶数门则保持空白。

- `Input structure`：
  - 背景为Black。
  - 若干种颜色的单像素宽线条将网格划分为不重叠的矩形房间。
  - 每个房间的四面墙壁颜色相同（由单一颜色围成）。
  - 房间边框上可能有单像素缺口（门），缺口处为Black。
  - 每个房间内部初始全为Black。

- `Rule`：
  - 对每个房间，统计其边框上Black缺口（门）的总数。
  - 若门的数量为奇数：用该房间的边框颜色填充整个房间内部（不改变边框本身和门缺口）。
  - 若门的数量为偶数（含0）：房间内部保持Black不变。
  - 所有边框像素和门缺口均保持原状。

- `Key insight`：做题者第一反应会关注房间的大小、位置、边框颜色差异等视觉显著属性来决定是否填充。实际的决定因素是"门的数量的奇偶性"——一个需要逐个房间仔细计数边框缺口的微妙条件。难点在于：门（缺口）在视觉上极不起眼，是墙壁上缺失的单个像素，容易被当作绘图瑕疵而非关键信息。人类需要从"墙壁的缺失"这个否定性特征中提取计数信息，并注意到奇偶性而非具体数值才是分类标准。

验证样例（颜色首字母：K=Black, R=Red, B=Blue, G=Green）：

Input:
```
R R R R R K G G G G G
R K K K R K G K K K G
R K K K R K G K K K G
R R K R R K G G G G G
K K K K K K K K K K K
B B B B B K K K K K K
B K K K B K K K K K K
B K K K B K K K K K K
B B B B B K K K K K K
```

Output:
```
R R R R R K G G G G G
R R R R R K G K K K G
R R R R R K G K K K G
R R K R R K G G G G G
K K K K K K K K K K K
B B B B B K K K K K K
B K K K B K K K K K K
B K K K B K K K K K K
B B B B B K K K K K K
```

简要说明此样例如何体现规则：Red房间(行0-3,列0-4)边框检查——行0列0-4: R R R R R(完整)；列4行0-3: R R R R(完整)；行3列0-4: R R K R R(列2处有1个Black缺口=1个门)；列0行0-3: R R R R(完整)。总门数=1(奇数)→用Red填充内部。内部像素(1,1)(1,2)(1,3)(2,1)(2,2)(2,3)全填Red，门缺口(3,2)保持Black。Green房间(行0-3,列6-10)边框检查——行0列6-10: G G G G G(完整)；列10行0-3: G G G G(完整)；行3列6-10: G G G G G(完整)；列6行0-3: G G G G(完整)。所有墙完整，无缺口。总门数=0(偶数)→内部保持Black。Blue房间(行5-8,列0-4)边框检查——所有墙完整。总门数=0(偶数)→内部保持Black。
</new_puzzle>

<new_puzzle>
- `description`：每个彩色物体都有一个同色的远方孤立锚点，物体沿锚点反方向投射Grey影子，影子长度等于物体包围盒短边的像素数。

- `Input structure`：
  - 背景为Black的网格。
  - 若干由不同前景色构成的小型4-连通物体（2-5像素）。
  - 对每种前景色，网格中另有一个同色孤立单像素（"锚点"），与该色物体不4-连通相邻（至少间隔1个Black像素）。
  - 锚点相对于物体质心在上、下、左、右四个轴向之一（行差绝对值与列差绝对值中有一个严格大于另一个，取较大者确定主轴方向）。

- `Rule`：
  - 对每个前景物体，找到同色锚点。
  - 确定主轴方向：比较锚点与物体质心的行差绝对值和列差绝对值，较大者为主轴。若锚点在物体上方则主轴方向=上，下方=下，左方=左，右方=右。
  - 影子投射方向 = 锚点相对于物体质心的反方向（锚点在上→影子朝下，锚点在左→影子朝右，以此类推）。
  - 影子长度L = 物体包围盒的min(行数, 列数)。
  - 对物体沿投射方向的最外侧边缘的每个像素，沿投射方向延伸L个Grey像素。
  - 物体本身和锚点保持不变。Grey像素不覆盖已有非Black像素。

- `Key insight`：做题者会自然地用物体到网格边界的距离或物体面积来解释影子长度，用物体形状来推测影子方向。但方向取决于一个远方同色孤立像素的位置（且是反方向投射），长度取决于包围盒的短边——这两个信息来源都是非直觉的。孤立的同色像素容易被当作噪点忽略，而"包围盒短边"不是人类自然关注的属性。做题者需要同时发现远方锚点的方向控制功能和包围盒短边的长度编码功能。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K
K K K R K K K K
K K K K K K K K
K R R K K K K K
K R R K K B K K
K K K K K B B K
K K K K K K K K
K K K K B K K K
```

Output:
```
K K K K K K K K
K K K R K K K K
K K K K K E K K
K R R K K E K K
K R R K K B K K
K R R K K B B K
K R R K K K K K
K K K K B K K K
```

简要说明此样例如何体现规则：Red物体在(3,1)(3,2)(4,1)(4,2)，包围盒2x2，min=2。Red锚点在(1,3)。物体质心=(3.5,1.5)，锚点=(1,3)。行差=|1-3.5|=2.5，列差=|3-1.5|=1.5。行差>列差→主轴=垂直。锚点在上方→影子朝下。物体下边缘=行4，列1和列2。每个像素向下延伸2个Grey：(5,1)=E, (6,1)=E, (5,2)=E, (6,2)=E。但(5,1)和(5,2)在原Input中均为K→可填Grey。注意输出中(5,1)(5,2)(6,1)(6,2)应为E，但我在Output中写成了R，让我修正——影子颜色是Grey不是物体颜色。修正后Output中(5,1)=R和(6,1)=R应该是E。

修正后的Output需重新检查：实际规则中影子像素颜色为Grey(E)。但Output我写为R R(物体颜色)。实际上物体像素(3,1)(3,2)(4,1)(4,2)保持Red，影子(5,1)(5,2)(6,1)(6,2)应为Grey。让我写出正确的Output：

修正Output:
```
K K K K K K K K
K K K R K K K K
K K K K K E K K
K R R K K E K K
K R R K K B K K
K E E K K B B K
K E E K K K K K
K K K K B K K K
```

Blue物体在(4,5)(5,5)(5,6)，包围盒行4-5列5-6即2x2，min=2。Blue锚点在(7,4)。物体质心≈(4.67,5.33)，锚点=(7,4)。行差=|7-4.67|=2.33，列差=|4-5.33|=1.33。行差>列差→主轴=垂直。锚点在下方→影子朝上。物体上边缘=行4，仅(4,5)处有Blue。向上延伸2个Grey：(3,5)=E, (2,5)=E。正确。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干个由框线围成的矩形区域，框线的四面壁厚度不均匀（1或2像素宽）。壁厚度为2的方向标记为"活跃方向"，框内的种子像素仅向活跃方向射出直线。

- `Input structure`：
  - 背景为Black。
  - 若干个由同色像素围成的封闭矩形框架，彼此不重叠。
  - 每个框架的四面墙（上、下、左、右）厚度分别为1像素或2像素宽。至少有一面墙厚度为2。
  - 每个框架内部初始为Black，且恰好包含一个非框架色、非Black的单像素种子。

- `Rule`：
  - 对每个框架，检测四面墙的厚度。厚度为2的墙所对应的方向即为"活跃方向"。
  - 从种子像素出发，向每个活跃方向画一条直线（用种子的颜色），直到碰到框架的内壁边缘（不覆盖框架像素）。
  - 厚度为1的方向不画线。
  - 种子像素保留，框架不变。

- `Key insight`：做题者的注意力会集中在种子像素的位置和颜色上，试图从种子与框架的空间关系中推导射线方向。但真正决定射线方向的信息藏在框架本身的微观结构中——四面墙厚度的差异。人类倾向于将框架视为均匀的边界，很难注意到"某面墙是2像素厚而另一面只有1像素厚"这种微妙的不对称性。核心认知转换在于：从关注"内容"（种子）转向关注"容器"（框架）的不对称几何属性。

验证样例（颜色首字母：K=Black, E=Grey, R=Red, G=Green, B=Blue）：

Input:
```
K K K K K K K K K K
K E E E E E E K K K
K E E E E E E K K K
K E E K K K E K K K
K E E K R K E K K K
K E E K K K E K K K
K E E E E E E K K K
K K K K K K K K K K
K K B B B B B B K K
K K B K K K K B K K
K K B K K K K B K K
K K B K K G K B K K
K K B K K K K B K K
K K B B B B B B K K
K K B B B B B B K K
K K K K K K K K K K
```

Output:
```
K K K K K K K K K K
K E E E E E E K K K
K E E E E E E K K K
K E E K R K E K K K
K E E R R K E K K K
K E E K K K E K K K
K E E E E E E K K K
K K K K K K K K K K
K K B B B B B B K K
K K B K K K K B K K
K K B K K K K B K K
K K B K K G K B K K
K K B K K G K B K K
K K B B B B B B K K
K K B B B B B B K K
K K K K K K K K K K
```

简要说明此样例如何体现规则：

第一个框架（Grey色E）墙壁厚度分析：上墙=行1-2(列1-6，两行全为E)→厚度2(活跃)；下墙=行6(列1-6，单行E)→厚度1(不活跃)；左墙=列1-2(行1-6，两列全为E)→厚度2(活跃)；右墙=列6(行1-6，单列E)→厚度1(不活跃)。活跃方向=上和左。内部空间行3-5、列3-5。Red种子在(4,4)。向上画线：(3,4)填R。向左画线：(4,3)填R。

第二个框架（Blue色B）墙壁厚度分析：上墙=行8(列2-7，单行B)→厚度1(不活跃)；下墙=行13-14(列2-7，两行全为B)→厚度2(活跃)；左墙=列2(行8-14，单列B)→厚度1(不活跃)；右墙=列7(行8-14，单列B)→厚度1(不活跃)。活跃方向=仅下方。内部空间行9-12、列3-6。Green种子在(11,5)。向下画线：(12,5)填G。
</new_puzzle>
