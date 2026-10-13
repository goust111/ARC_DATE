<new_puzzle>
- `description`：网格由均匀的Grey墙壁划分成若干房间，每个房间内有一个彩色标记像素；相邻房间标记颜色相同时，拆除它们之间的共享墙壁段。

- `Input structure`：
  - 整个网格被1像素宽的Grey墙壁均匀划分为m行n列的等大矩形房间，外边界也是Grey墙壁。
  - 每个房间内部为Black背景，恰好包含一个非Black非Grey的彩色单像素标记。
  - 标记颜色从2-3种颜色中取值。

- `Rule`：
  - 对每对水平或垂直相邻的房间，比较它们内部标记的颜色。
  - 若两个相邻房间的标记颜色相同，则将它们之间的共享墙壁段变为Black。共享墙壁段定义为仅属于这两个房间之间的那一排或列Grey像素，不包含墙壁交叉的节点像素（即与其他墙壁共享的十字或T型交汇点）。
  - 若标记颜色不同，共享墙壁保持Grey不变。
  - 外边界墙壁永远保持Grey。
  - 标记像素本身保留不变。

- `Key insight`：做题者看到房间里的彩色标记，最先倾向于对标记本身做操作——连线、移动或重着色。但实际被变换的对象是通常被视为静态背景结构的墙壁。核心难点在于"角色反转"：标记仅是判断条件的输入，墙壁才是被操作的主体。做题者需要将注意力从前景标记转移到背景结构上。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
E E E E E E E E E
E K B K E K B K E
E K K K E K K K E
E K K K E K K K E
E E E E E E E E E
E K R K E K B K E
E K K K E K K K E
E K K K E K K K E
E E E E E E E E E
```

Output:
```
E E E E E E E E E
E K B K K K B K E
E K K K K K K K E
E K K K K K K K E
E E E E E K K K E
E K R K E K B K E
E K K K E K K K E
E K K K E K K K E
E E E E E E E E E
```

此样例为2x2房间网格，每房间内部3x3。左上房间标记Blue(1,2)，右上房间标记Blue(1,6)，左下房间标记Red(5,2)，右下房间标记Blue(5,6)。左上与右上同色Blue→拆除列4行1-3的墙壁（3个Grey像素变Black）。右上与右下同色Blue→拆除行4列5-7的墙壁（3个Grey像素变Black）。左上与左下B!=R→保留。左下与右下R!=B→保留。行4列4是墙壁交叉节点，保持Grey。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干条从边界种子出发的彩色射线互相交叉；交叉点显示哪种颜色由网格角落的优先级色条决定——色条中排列越靠右的颜色优先级越高，在交叉处覆盖优先级低的颜色。

- `Input structure`：
  - 背景为Black。
  - 网格第0行最左侧连续若干列组成"优先级色条"，包含2-4个不同颜色的单像素，从左到右排列。
  - 网格的上、下、左、右边界上散布若干彩色单像素种子（不与色条重叠），每个种子的颜色在色条中出现。
  - 每个种子沿垂直于其所在边界的方向向网格内部发射射线（上边界→向下，下边界→向上，左边界→向右，右边界→向左）。

- `Rule`：
  - 从每个种子出发，沿垂直于边界的方向画一条直线，延伸到网格对面边界。
  - 当两条射线在同一像素位置交叉时，保留色条中位置更靠右（优先级更高）的颜色。
  - 种子像素本身始终保持原色，不被其他射线覆盖。
  - 色条在输出中保留不变。

- `Key insight`：做题者看到交叉的射线，会尝试从颜色本身的某种固有属性推断优先级（如水平总在垂直之上，或"暖色压冷色"），但优先级完全由角落色条这个容易被忽略的"装饰性"元素编码。核心难点在于发现色条不是标题或装饰，而是解码交叉覆盖关系的外部字典，且这个字典与颜色本身的任何视觉属性无关。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
R B G K G K K
K K K K K K K
K K K K K K K
K K K K K K B
K K K K K K K
K K K K K K K
R K K K K K K
```

Output:
```
R B G K G K K
R K K K G K K
R K K K G K K
R B B B G B B
R K K K G K K
R K K K G K K
R K K K G K K
```

色条(0,0)-(0,2)：R, B, G，优先级R<B<G。种子：G在(0,4)上边界→向下射线覆盖列4行1-6。B在(3,6)右边界→向左射线覆盖行3列0-5。R在(6,0)左边界→向上射线覆盖列0行0-5。交叉(3,0)：R与B交叉，B优先→Blue。交叉(3,4)：B与G交叉，G优先→Green。列0行1-2和行4-5仅R射线→Red。行3列1-3和列5仅B射线→Blue。列4行1-2和行4-6仅G射线→Green。
</new_puzzle>

<new_puzzle>
- `description`：网格左上角有一个Grey像素组成的小型非对称图案作为"橡皮擦模板"；对网格中每个彩色实心形状，将模板左上角对齐到该形状包围盒左上角，模板覆盖到的该形状像素被擦除为背景色。

- `Input structure`：
  - 背景为Black。
  - 网格左上角区域有一个由Grey像素组成的小型非对称图案（橡皮擦模板）。
  - 网格其余区域有若干个不同颜色的连通实心形状（如矩形），形状之间互不相邻。

- `Rule`：
  - 提取Grey模板的形状。以模板包围盒的左上角为锚点，记录所有Grey像素相对于锚点的偏移。
  - 对每个彩色连通形状，计算其包围盒的左上角坐标。
  - 将模板锚点对齐到该形状包围盒的左上角，得到模板各像素的绝对坐标。
  - 若某个模板像素的绝对坐标落在该彩色形状的像素上，将该位置擦除为Black。
  - 落在背景或其他颜色上的模板像素不做任何操作。
  - 每个形状独立应用模板。
  - Grey模板本身在输出中被擦除（变为Black）。

- `Key insight`：做题者看到Grey图案，最先猜测它是一个"印章"——要被复制到某处添加新内容。但Grey图案实际是"橡皮擦"，定义了要从每个形状中删去的部分。这种从"加法思维"到"减法思维"的反转是核心难点。另一个易错点是做题者可能以为模板对齐到形状的中心，但实际对齐的是包围盒左上角。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
E E K K K K K K K
K E K K K K K K K
K K K K K K K K K
K K R R R K K K K
K K R R R K K K K
K K R R R K K K K
K K K K K B B B K
K K K K K B B B K
K K K K K B B B K
```

Output:
```
K K K K K K K K K
K K K K K K K K K
K K K K K K K K K
K K K K R K K K K
K K K R R K K K K
K K R R R K K K K
K K K K K K K B K
K K K K K K B B K
K K K K K B B B K
```

Grey模板像素：(0,0)(0,1)(1,1)，形成倒L形。偏移集合：(0,0)(0,1)(1,1)。Red形状(3,2)-(5,4)为3x3实心块，包围盒左上角(3,2)。模板对齐后擦除位置：(3,2)(3,3)(4,3)，这三个位置都是Red→变Black。Red剩余6个像素：(3,4)(4,2)(4,4)(5,2)(5,3)(5,4)。Blue形状(6,5)-(8,7)为3x3实心块，包围盒左上角(6,5)。擦除位置：(6,5)(6,6)(7,6)→变Black。Blue剩余6个像素：(6,7)(7,5)(7,7)(8,5)(8,6)(8,7)。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干彩色实心矩形，每个矩形恰好有一条边上开了一个单像素缺口；矩形整体被清空为背景色，然后从缺口位置向外射出一条单像素宽的同色射线，直到碰到网格边界或其他非背景像素。

- `Input structure`：
  - 背景为Black。
  - 若干个不同颜色的实心矩形（完全填充，至少3x3大小）。
  - 每个矩形恰好有一条边上存在一个单像素缺口（该边上有一个像素被替换为Black）。
  - 缺口不在边的两端（即缺口两侧至少各有一个该色像素）。

- `Rule`：
  - 对每个矩形，找到缺口所在的边（上、下、左、右）和缺口的列或行位置。
  - 缺口所在的边指示射出方向：上边有缺口→向上射出，下边→向下，左边→向左，右边→向右。
  - 原矩形的所有非缺口像素全部变为Black（矩形被清空）。
  - 从缺口位置紧邻矩形外侧开始，沿射出方向画一条单像素宽的同色射线，直到碰到网格边界或其他非背景像素前一格为止。

- `Key insight`：做题者看到缺口矩形，最可能猜测矩形沿缺口方向整体平移。但实际操作是矩形被销毁，仅从缺口处射出一条细线。关键的认知跳跃在于：(1) 矩形不是被移动而是被清空；(2) 缺口不仅编码方向，还精确定位了射线在哪一列或行；(3) 输出中原矩形区域完全变空，存留的只有一条从缺口向外延伸的细线。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K
K K K K K K K K
K R R R R K K K
K R R R R K K K
K R R R R K K K
K R K R R K K K
K K K K K K K K
K K K K K K K K
```

Output:
```
K K K K K K K K
K K K K K K K K
K K K K K K K K
K K K K K K K K
K K K K K K K K
K K K K K K K K
K K R K K K K K
K K R K K K K K
```

Red矩形占据(2,1)-(5,4)。底边第5行像素为：(5,1)=R, (5,2)=K(缺口), (5,3)=R, (5,4)=R。缺口在底边列2处→向下射出。矩形所有R像素清空为Black。从(6,2)开始向下画Red射线到网格底部边界：(6,2)和(7,2)变Red。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干对同色单像素标记，每对标记定义一个矩形区域的两个对角；输出中每个矩形被填充为对应颜色，重叠区域由网格末行的优先级序列决定显示哪种颜色。

- `Input structure`：
  - 背景为Black。
  - 若干对同色单像素标记散布在网格中，每种非背景颜色恰好出现两个像素。
  - 每对同色像素定义一个矩形区域：以两像素为对角的最小轴对齐矩形。
  - 网格最后一行有一个优先级序列：从左到右排列所有标记颜色各一个像素，间隔一格Black。

- `Rule`：
  - 对每种颜色，找到该颜色的两个标记像素，计算以它们为对角的矩形。
  - 读取最后一行的优先级序列，从左到右优先级递增（最右=最高优先级）。
  - 按优先级从低到高依次填充每个矩形区域为对应颜色。
  - 高优先级矩形的像素覆盖低优先级矩形的像素。
  - 标记像素和优先级序列保持不变。

- `Key insight`：做题者看到成对出现的同色点，最可能猜测它们是"画线的端点"或"移动指令"。但两个点实际定义的是一个矩形区域的对角，这种从"点对"到"面积"的概念跳跃是第一个难点。第二个难点是重叠时的优先级由末行色块序列编码，与颜色本身的属性无关，做题者需要发现这个分离的序列并理解其含义。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K
R K K K G K K
K K K K K K K
K K K K K K K
K K G K K K R
K K K K K K K
R K G K K K K
```

Output:
```
K K K K K K K
R R G G G R R
R R G G G R R
R R G G G R R
R R G G G R R
K K K K K K K
R K G K K K K
```

优先级序列在第6行：(6,0)=R, (6,2)=G，优先级R<G。Red两个标记(1,0)和(4,6)定义矩形行1-4列0-6。Green两个标记(1,4)和(4,2)定义矩形行1-4列2-4。先填Red（低优先级）：行1-4列0-6全变Red。再填Green（高优先级）：行1-4列2-4覆盖变Green。最终行1-4中列0-1为Red，列2-4为Green，列5-6为Red。标记像素已是对应颜色，保持不变。
</new_puzzle>

<new_puzzle>
- `description`：网格中散布若干个相同大小的单色实心方块和一个独特的双色方块（指令方块）；每个单色方块将自身变为指令方块的图案副本——指令方块中与该方块同色的位置保持原色，非同色位置改为指令方块中该位置的颜色。

- `Input structure`：
  - 背景为Black。
  - 一个NxN的"指令方块"由两种颜色A和B组成，形成一个二色图案。
  - 若干个NxN的单色实心方块散布在网格中，颜色为A或B之一。
  - 指令方块是唯一包含两种颜色的方块，以此与单色方块区分。

- `Rule`：
  - 识别指令方块及其两种颜色A和B在各像素位置的分布。
  - 对每个单色方块，将其变为指令方块的图案副本：逐像素复制指令方块的颜色模式。
  - 即所有单色方块在输出中都变成了与指令方块完全相同的二色图案。
  - 指令方块本身在输出中保持不变。

- `Key insight`：做题者看到多个同大方块，会先尝试寻找它们之间的排序、移动或比较规则。双色方块因为"不同"而容易被当作异常值或干扰项忽略。核心难点在于识别出双色方块是"模板"，而所有单色方块都需要被改造成它的样子。做题者习惯于从多个相似对象中提取共性规则，但这里规则来自一个独特的例外对象。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K
K R B K K K K K K
K R R K K K K K K
K K K K K K K K K
K K K K R R K K K
K K K K R R K K K
K K K K K K K K K
K K K K K B B K K
K K K K K B B K K
```

Output:
```
K K K K K K K K K
K R B K K K K K K
K R R K K K K K K
K K K K K K K K K
K K K K R B K K K
K K K K R R K K K
K K K K K K K K K
K K K K K R B K K
K K K K K R R K K
```

指令方块在(1,1)-(2,2)，图案为：(0,0)=R, (0,1)=B, (1,0)=R, (1,1)=R。Red单色方块(4,4)-(5,5)原为全Red，变为指令图案副本：(4,4)=R, (4,5)=B, (5,4)=R, (5,5)=R。Blue单色方块(7,5)-(8,6)原为全Blue，变为指令图案副本：(7,5)=R, (7,6)=B, (8,5)=R, (8,6)=R。所有单色方块输出后与指令方块图案完全一致。
</new_puzzle>
