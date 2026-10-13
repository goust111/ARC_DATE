<new_puzzle>
- `description`：网格中有若干同心嵌套的矩形边框，输出时将各层边框的颜色按嵌套深度循环内移一层——最内层获得最外层的颜色，其余每层获得比自己内一层的原始颜色。

- `Input structure`：
  - 背景为Black。
  - 网格中有N个（N≥2）同心嵌套的一像素宽矩形边框，从外到内颜色各不相同。
  - 相邻两层边框之间间隔至少一像素的背景色。
  - 最内层边框内部为背景色（空心）。

- `Rule`：
  - 按从外到内的顺序将边框标记为第1层、第2层、…、第N层。
  - 颜色循环内移一层：第1层获得第2层的颜色，第2层获得第3层的颜色，…，第N-1层获得第N层的颜色，第N层获得第1层的颜色。
  - 所有边框的几何位置和形状完全不变，仅颜色发生循环。
  - 背景保持不变。

- `Key insight`：做题者看到同心矩形时，最自然的思路是寻找几何变换（缩放、平移、增删层级）。颜色在层间的循环流动是纯属性重映射，与空间结构完全解耦。进一步的难点在于确定循环方向（向内还是向外移动），需要同时对比多个样例中所有层的颜色变化才能确认。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
R R R R R R R R R
R K K K K K K K R
R K G G G G G K R
R K G K K K G K R
R K G K B K G K R
R K G K K K G K R
R K G G G G G K R
R K K K K K K K R
R R R R R R R R R

Output:
G G G G G G G G G
G K K K K K K K G
G K B B B B B K G
G K B K K K B K G
G K B K R K B K G
G K B K K K B K G
G K B B B B B K G
G K K K K K K K G
G G G G G G G G G

简要说明：三层嵌套边框从外到内原为Red(1)、Green(2)、Blue(3)。循环内移一层后：第1层→第2层颜色Green，第2层→第3层颜色Blue，第3层→第1层颜色Red。输出：最外层Green、中层Blue、最内层Red。
</new_puzzle>

<new_puzzle>
- `description`：网格中有一条Grey蜿蜒路径和路径侧边的若干彩色单像素标记；输出时路径上每个像素被染成沿路径拓扑步数最近的标记颜色，而非空间直线距离最近的颜色。

- `Input structure`：
  - 背景为Black。
  - 一条由Grey像素构成的单像素宽连续路径蜿蜒穿过网格，路径不分叉、不自交、仅沿正交方向延伸。
  - 路径侧边（与路径上某像素正交相邻但自身不在路径上）散布着若干不同颜色的单像素标记，每个标记颜色唯一且不为Grey或Black。
  - 每个标记恰好与路径上的某一个像素正交相邻。

- `Rule`：
  - 将路径视为从一端到另一端的有序一维像素序列。路径起点定义为端点中行号更小者（行号相同则列号更小者）。
  - 每个标记通过其唯一邻接的路径像素在序列中获得一个索引。
  - 路径上每个像素被染成沿路径步数最近的标记的颜色；若到两个标记距离相等，取索引更小的标记颜色。
  - 标记像素本身在输出中保留不变。

- `Key insight`：做题者看到路径和旁边的彩色标记，最直觉的反应是用空间直线距离分配颜色。但当路径蜿蜒折返时，空间上很近的标记在路径拓扑上可能反而很远。核心洞见是"距离"须沿路径度量——这是对距离概念的认知重构，而非简单的信息提取。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
E E E K K
R K E K K
K K E E E
K K B K E
K K K K E

Output:
R R R K K
R K B K K
K K B B B
K K B K B
K K K K B

简要说明：路径序列为(0,0)→(0,1)→(0,2)→(1,2)→(2,2)→(2,3)→(2,4)→(3,4)→(4,4)，索引0至8。Red在(1,0)邻接(0,0)索引0。Blue在(3,2)邻接(2,2)索引4。索引0距Red=0距Blue=4→Red；索引1距Red=1距Blue=3→Red；索引2距Red=2距Blue=2→等距取索引小的Red；索引3距Red=3距Blue=1→Blue；索引4-8全距Blue更近→Blue。
</new_puzzle>

<new_puzzle>
- `description`：网格中有一个Grey骨架框（矩形外框+十字内隔线），框线上嵌有若干不起眼的单像素色标；框外散布着彩色碎片。碎片不是按自身空间位置、而是按颜色匹配框线上色标来决定放入哪个分区。

- `Input structure`：
  - 背景为Black。
  - 网格中有一个由Grey像素绘制的矩形外框，内部被同色十字线（一条水平线+一条垂直线）均分为四个等大分区（2×2布局）。
  - 十字线的各段上有1-4个像素被替换为非Grey非Black的颜色（色标），每个色标位于某个分区的边界上，标识该分区。色标颜色互不相同。
  - 框外散布着1-4个小型单色碎片，每个碎片颜色恰好匹配某个色标颜色。

- `Rule`：
  - 每个色标标记了它所在边界所属的分区。
  - 框外碎片按颜色匹配色标：碎片颜色等于哪个色标颜色，就放入该色标所属的分区。
  - 碎片在目标分区内左上角对齐放置。
  - 框外碎片原位置清除为背景色；框线（含色标）保持不变。
  - 若某分区无匹配碎片，该分区保持背景色。

- `Key insight`：做题者第一反应是按碎片的空间位置（在框的上下左右哪一侧）决定放入哪个分区——这在大多数ARC题中是合理的直觉。但本题中碎片位置与目标分区无关，匹配键是颜色。做题者需要注意到框线上容易被当作噪声忽略的色标像素，并发现颜色匹配才是分配规则。这是典型的"看似装饰的边框元素实为核心规则载体"的角色反转。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
K K K K K K K K K
K O O K K K K K K
K O K K K K K K K
E E E E O E E E E
E K K E K K K K E
E K K E K K K K E
E E B E E E E E E
E K K K E K K K E
E K K K E K K K E
E E E E E E E E E
K K K K K K B B K
K K K K K K K K K

Output:
K K K K K K K K K
K K K K K K K K K
K K K K K K K K K
E E E E O E E E E
E K K E O O K K E
E K K E O K K K E
E E B E E E E E E
E B B K E K K K E
E K K K E K K K E
E E E E E E E E E
K K K K K K K K K
K K K K K K K K K

简要说明：Grey骨架框行3-9列0-8，十字线行6（水平）和列4（垂直）分四个分区：左上(行4-5,列1-3)、右上(行4-5,列5-7)、左下(行7-8,列1-3)、右下(行7-8,列5-7)。色标Orange在(3,4)位于右上分区上边界，色标Blue在(6,2)位于左下分区上边界。框外Orange碎片(1,1)(1,2)(2,1)颜色匹配色标O→放入右上分区左上角对齐。框外Blue碎片(10,6)(10,7)颜色匹配色标B→放入左下分区左上角对齐。碎片原位清除。
</new_puzzle>

<new_puzzle>
- `description`：网格中每个彩色矩形对象内部有一条贯穿的背景色"裂缝线"和一个Grey单像素标记。输出时，无标记的那一半沿裂缝平行方向滑动，滑动量等于标记到裂缝的正交距离。

- `Input structure`：
  - 背景为Black。
  - 网格中有若干个矩形单色区域（不同对象颜色不同），对象之间互不接触。
  - 每个对象内部恰好有一条贯穿的背景色直线（水平或垂直），将对象分为大小相等的两半。
  - 每个对象的其中一半内有恰好一个Grey单像素标记。
  - 标记到裂缝线的正交距离为d（d≥1）。

- `Rule`：
  - 对每个对象，找到其内部裂缝线的方向和Grey标记的位置。
  - 含标记的那一半保持不动。
  - 另一半（无标记半）沿裂缝的平行方向滑动d个像素。具体地：若裂缝为水平线，无标记半向右滑动d格；若裂缝为垂直线，无标记半向下滑动d格。
  - 滑动后原位置露出的像素变为背景色。
  - 标记像素在输出中保持Grey色不变。

- `Key insight`：做题者容易关注裂缝本身的位置或方向作为变换的决定因素，或以为两半做简单的分离/镜像操作。真正的规则中，滑动量来自一个容易被忽视的单像素标记与裂缝的距离——这是"标记的位置属性间接控制另一组件行为"的跨组件信息编码。做题者还需要发现滑动方向（沿裂缝方向而非垂直于裂缝）的非直觉性。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
K K K K K K K K K
K R R R R R K K K
K R E R R R K K K
K K K K K K K K K
K R R R R R K K K
K R R R R R K K K
K K K K K K K K K

Output:
K K K K K K K K K
K R R R R R K K K
K R E R R R K K K
K K K K K K K K K
K K R R R R R K K
K K R R R R R K K
K K K K K K K K K

简要说明：Red对象占据行1-2和行4-5（列1-5），行3是贯穿的Black裂缝线（水平）。Grey标记在(2,2)位于上半部分，到裂缝(行3)的距离d=1。上半含标记不动，下半沿裂缝平行方向（水平）向右滑动1格：从列1-5移至列2-6。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干单色碎片，每个碎片旁有一个Grey单像素标记；输出时以Grey标记为中心将碎片做180度点对称旋转产生副本，原件保留不变。

- `Input structure`：
  - 背景为Black。
  - 网格中有若干个小型（2-4像素）单色4-连通碎片，颜色各不相同。
  - 每个碎片旁有恰好一个Grey单像素标记，与碎片的某个像素正交相邻但不属于碎片。
  - Grey标记与碎片一一对应。
  - 碎片之间（含副本区域）互不重叠。

- `Rule`：
  - 对每个碎片，将其关联的Grey标记视为旋转中心。
  - 碎片的每个像素(r,c)关于中心(cr,cc)做180度旋转映射到(2*cr-r, 2*cc-c)。
  - 在旋转后的位置用碎片的原色绘制副本。
  - 原件、Grey标记和背景均保持不变。

- `Key insight`：做题者可能以为Grey标记是方向指示器（标记在碎片哪一侧就往哪个方向复制），或尝试寻找平移/轴对称镜像规则。但只有将Grey视为旋转中心做180度点对称旋转时，副本的位置和朝向才能被统一解释。当碎片是非对称形状（如L形）时，副本的朝向翻转是判断旋转而非平移/镜像的关键线索。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
K K K K K K K
K R R K K K K
K K R K K K K
K K E K K K K
K K K K K K K
K K K K K K K
K K K K K K K

Output:
K K K K K K K
K R R K K K K
K K R K K K K
K K E K K K K
K K R K K K K
K R R K K K K
K K K K K K K

简要说明：Red碎片占据(1,1)(1,2)(2,2)。Grey标记在(3,2)。以(3,2)为中心180度旋转：(1,1)→(5,3)，(1,2)→(5,2)，(2,2)→(4,2)。副本Red出现在(4,2)(5,2)(5,3)。原件是朝右上的L形，副本是朝左下的L形，体现了点对称。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干"窗户"图案（矩形边框内含等间距同色档条），输出时档条方向旋转90度——水平档变垂直档——而矩形外框保持不变。

- `Input structure`：
  - 背景为Black。
  - 网格中有若干个由单一颜色绘制的一像素宽矩形闭合边框。
  - 每个边框内部包含若干条与边框同色的等间距直线档条，连接对应的两条平行边框边。
  - 档条要么全是水平的（连接左右边），要么全是垂直的（连接上下边）。

- `Rule`：
  - 对每个窗户，保持矩形边框不变。
  - 擦除原有档条，在垂直于原方向上绘制相同数量的新档条。
  - 新档条等间距分布，连接对应边框边，颜色与边框相同。

- `Key insight`：做题者初看可能以为整个窗户被旋转了90度，但边框的宽高比没有变化——只有内部档条方向改变了。做题者需要将"外框"和"内部结构"识别为两个独立层次，并发现变换仅作用于内部而保持容器不变。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
R R R R R
R K K K R
R R R R R
R K K K R
R R R R R

Output:
R R R R R
R K R K R
R K R K R
R K R K R
R R R R R

简要说明：Red窗户5×5，内部3×3。原有1条水平档条（行2连接列1-3），分内部为上下两区。输出改为1条垂直档条（列2连接行1-3），分内部为左右两区。边框形状不变。
</new_puzzle>

<new_puzzle>
- `description`：网格中有一个由Grey边框围成的封闭矩形区域，内部散布着2-4个不同颜色的单像素锚点。输出时内部每个背景色像素被染成曼哈顿距离最近的锚点的颜色，形成Voronoi划分。

- `Input structure`：
  - 背景为Black。
  - 网格中有一个由Grey像素构成的封闭矩形轮廓（1像素宽）。
  - 轮廓严格内部有2-4个不同颜色的单像素锚点。
  - 轮廓内部其余像素为背景色。轮廓外部全为背景色。

- `Rule`：
  - Grey轮廓像素保持不变。
  - 轮廓外部像素保持背景色不变。
  - 对轮廓内部每个背景色像素，计算到每个锚点的曼哈顿距离（|行差|+|列差|）。
  - 将该像素染成距离最近锚点的颜色。若距离相等，取行号较小的锚点；行号同则取列号较小的。
  - 锚点本身保留原色。

- `Key insight`：做题者看到锚点和输出中的彩色区域，最初可能以为是洪泛填充或基于连通性的扩散。但区域边界呈现的斜向锯齿形状无法用洪泛解释。核心认知转换是：从"颜色从锚点扩散"的过程性思维转向"每个像素独立选择最近锚点"的声明式思维。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
K K K K K K K K
K E E E E E E K
K E K K K K E K
K E K B K K E K
K E K K K K E K
K E K K K R E K
K E K K K K E K
K E E E E E E K
K K K K K K K K

Output:
K K K K K K K K
K E E E E E E K
K E B B B B E K
K E B B B B E K
K E B B B R E K
K E B B R R E K
K E B B R R E K
K E E E E E E K
K K K K K K K K

简要说明：内部5×5区域(行2-6,列2-6)。锚B在(3,3)，锚R在(5,5)。逐像素曼哈顿距离计算：(2,2)B=2,R=6→B；(2,5)B=3,R=3等距取行小B→B；(4,4)B=2,R=2等距取行小B→B；(4,5)B=3,R=1→R；(5,4)B=3,R=1→R；(6,4)B=4,R=2→R；(6,2)B=4,R=4等距取行小B→B。分界线呈斜向：B占据左上大部，R占据右下角。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干组水平条纹图案，每组中一行是完整的彩色模板、其余行是模板的带空洞副本。输出时将所有空洞还原为模板对应位置的颜色。

- `Input structure`：
  - 背景为Black。
  - 网格包含若干组水平条纹图案，组间由至少一行纯背景色隔开。
  - 每组中恰好有一行没有背景色像素（完整模板行），由多种颜色组成。
  - 同组其余行大部分像素与模板对应列颜色一致，但有若干像素被替换为背景色（空洞）。

- `Rule`：
  - 识别每组中唯一无空洞的完整行作为模板行。
  - 对同组每个非模板行，将背景色像素填充为模板行对应列的颜色。
  - 模板行和非空洞像素不变。组间分隔行不变。

- `Key insight`：做题者面对残缺行时，最初可能试图寻找缺损位置的空间规律。真正的洞见是组内存在一个"完整参考行"，其他行都是它的带缺损复制——规则是识别模板并填补，而非分析缺损模式。将思维从"找缺损规律"转换为"找完整样本并修复"是关键的视角转换。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
R G B R G
R K B K G
K G K R G
K K K K K
Y O Y O Y
Y O K O K
Y K Y K Y

Output:
R G B R G
R G B R G
R G B R G
K K K K K
Y O Y O Y
Y O Y O Y
Y O Y O Y

简要说明：第一组行0-2中行0"R G B R G"为完整模板，行1列1的K→G列3的K→R，行2列0的K→R列2的K→B。第二组行4-6中行4"Y O Y O Y"为模板，行5列2的K→Y列4的K→Y，行6列1的K→O列3的K→O。
</new_puzzle>

<new_puzzle>
- `description`：网格中有若干成对的同色单像素点，每对位于同一行或同一列。输出时在每对之间绘制连接线段，当两条线交叉时，交叉点颜色由绘制顺序决定：按各对较小坐标的字典序排列，后绘制的覆盖先绘制的。

- `Input structure`：
  - 背景为Black。
  - 网格中有若干种颜色，每种颜色恰好出现在两个位置（形成一对），每位置为单像素。
  - 每对同色像素位于同一行或同一列上。
  - 不同对的连接路径可能交叉。

- `Rule`：
  - 对每对同色像素，沿共享行或列绘制该颜色的连接线段（填充两端点之间所有中间像素）。
  - 绘制顺序：每对取坐标较小的端点(先比行,再比列)作为排序键，所有对按字典序升序排列，依次绘制。
  - 后绘制的线段在交叉点覆盖先绘制的。

- `Key insight`：做题者很快能发现"同色配对→连线"的基本规则，但交叉点颜色看似任意。做题者需要从多个样例中反向推断出交叉点颜色取决于隐含的全局绘制顺序——这个顺序编码在色块对的空间位置中。局部外观由全局排序约束决定是核心难点。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
K K K R K K
K K K K K K
B K K K K B
K K K K K K
K K K R K K
K K K K K K

Output:
K K K R K K
K K K R K K
B B B B B B
K K K R K K
K K K R K K
K K K K K K

简要说明：Red对在(0,3)和(4,3)同列，Blue对在(2,1)和(2,5)同行。排序键：Red(0,3)先于Blue(2,1)因行号0<2。先绘制Red垂直线填充(1,3)(2,3)(3,3)，再绘制Blue水平线填充(2,2)(2,3)(2,4)。交叉点(2,3)先被Red占、后被Blue覆盖，最终为Blue。
</new_puzzle>
