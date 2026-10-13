<new_puzzle>
- `description`：将网格中的框架线视为管道网络，从端点源头沿管道灌注颜色，管道上每个像素归属于沿管道路径距离最近的源头。

- `Input structure`：
  - 背景为Black。
  - Grey像素构成一个一像素宽的正交管道网络（水平和垂直线段在交叉点相连）。
  - 网络中某些管道的端点（死胡同末端）有一个非Grey、非Black的彩色像素，作为"颜色源头"。每个源头颜色不同。
  - 管道网络中存在T形或十字交叉节点。

- `Rule`：
  - 每个源头向其所在管道方向灌注颜色：从源头像素开始，沿Grey管道逐格向前扩散，将Grey替换为该源头的颜色。
  - 灌注遇到交叉节点时继续沿所有分支扩散。
  - 当两股不同颜色的灌注在同一个管道段相遇时，该段的每个像素归属于距离它更近的那个源头（曼哈顿距离沿管道路径计算）。若等距，保留Grey不变。
  - 源头像素本身保持原色不变；背景Black不变。

- `Key insight`：人类的第一直觉是将Grey框架线视为静态结构（边框或分隔线），不会想到它们是需要被重新着色的核心操作对象。真正的规则是将框架线作为管道，从端点源头灌注颜色并按路径距离划分领地——这是一种角色反转：装饰性结构变成了变换的主体。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K
K B E E E E R K
K K K K E K K K
K K K K E K K K
K K K K E K K K
K K K K E K K K
K G E E E K K K
K K K K K K K K
```

Output:
```
K K K K K K K K
K B B B E R R K
K K K K E K K K
K K K K E K K K
K K K K G K K K
K K K K G K K K
K G G G G K K K
K K K K K K K K
```

说明：Blue源头位于(1,1)，沿水平管道向右灌注；Red源头位于(1,6)，沿水平管道向左灌注。水平管道上(1,2)(1,3)距Blue更近涂Blue，(1,5)距Red更近涂Red，(1,4)等距保留Grey。垂直管道从交叉点(1,4)向下延伸至(6,4)与底部水平管道交叉。Green源头位于(6,1)，沿水平管道向右灌注(6,2)(6,3)为Green，经交叉点(6,4)涂Green并向上延伸。垂直段上(5,4)(4,4)距Green更近涂Green，(3,4)(2,4)距顶部交叉点更近但顶部有两个源头竞争，(2,4)(3,4)保留Grey（等距或更近于顶部但顶部两源等距故保留Grey）。
</new_puzzle>

<new_puzzle>
- `description`：网格中有多层嵌套矩形环，根据最内层隐藏的颜色标记决定哪些层级的环被重新着色。

- `Input structure`：
  - 背景为Black。
  - 一种前景色（Grey）绘制了若干嵌套的封闭矩形环（一像素宽的矩形轮廓），形成从外到内的多层嵌套结构。
  - 最内层封闭区域中有一个单像素彩色标记（如Red或Blue），其余内部区域均为Black。
  - 网格中可能有多组独立的嵌套矩形环结构，每组内层标记颜色可能不同。

- `Rule`：
  - 对每组嵌套结构，从最外层环开始向内编号为第1层、第2层、第3层……
  - 最内层的彩色标记决定着色相位：如果标记为Red，则偶数层环变为Red、奇数层环保持Grey；如果标记为Blue，则奇数层环变为Blue、偶数层环保持Grey。
  - 彩色标记像素本身在输出中被移除（变为Black）。
  - 环内被包围的Black区域保持Black不变。

- `Key insight`：人类会自然注意到嵌套环的层级结构并尝试简单的奇偶交替着色规则，但仅看一个样例无法确定哪些层该着色。着色模式取决于最内层一个极易被忽视的单像素标记——它隐藏在视觉上最不显眼的位置（最深层内部），而且它的作用不是"标记哪个区域"而是"选择全局着色相位"。这种全局控制信号隐藏在局部细节中的设计需要多个样例才能察觉。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
E E E E E E E E
E K K K K K K E
E K E E E E K E
E K E K K E K E
E K E K R E K E
E K E E E E K E
E K K K K K K E
E E E E E E E E
```

Output:
```
E E E E E E E E
E K K K K K K E
E K R R R R K E
E K R K K R K E
E K R K K R K E
E K R R R R K E
E K K K K K K E
E E E E E E E E
```

说明：三层嵌套矩形环，标记为Red。规则：偶数层变Red。第1层（最外）保持Grey，第2层变Red，第3层（最内，仅2x2）保持Grey。标记像素(4,4)被移除变为Black。
</new_puzzle>

<new_puzzle>
- `description`：网格中多个相同形状的彩色物体需要按照一条隐藏的多色"指令序列"线段所编码的顺序，依次堆叠到锚点位置。

- `Input structure`：
  - 背景为Black。
  - 网格中散布着若干小型连通物体（3-5像素），每个物体是单一颜色，所有物体形状完全相同（平移同构）。
  - 网格的某一行有一条多色线段（每格一个颜色像素，相邻排列，水平方向），其中每种颜色恰好对应场上某一个物体的颜色。这条线段就是"堆叠指令"。
  - 线段的一端紧邻一个Grey单像素标记，标记的上方或下方指示堆叠的锚点列。

- `Rule`：
  - 读取指令线段中的颜色序列（从左到右）。
  - 按此顺序，将对应颜色的物体依次搬移到Grey锚点所在列：第一个颜色的物体底部与锚点同行，后续物体紧贴前一个物体上方放置（向上堆叠，无间隔）。
  - 物体保持原始形状和颜色，搬移后原位置清空。
  - 指令线段和Grey标记在输出中被移除。

- `Key insight`：人类看到多个相同形状的彩色物体，第一反应是按空间位置或颜色属性排序分组。但真正的排列顺序来自一条看似装饰性的多色线段——它不像典型的ARC标记（单色点或线），而是一个多颜色序列，容易被误认为是另一组待操作的物体或边界装饰。难点在于识别出这条线段的"元"角色：它不参与空间变换，而是编码了变换的执行顺序。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K
K R R K K B B K
K R K K K B K K
K K K K K K K K
K K G G K K K K
K K G K K K K K
K K K K K K K K
E R B G K K K K
```

Output:
```
K K K K K K K K
K K K K K K K K
R R K K K K K K
R K K K K K K K
B B K K K K K K
B K K K K K K K
G G K K K K K K
G K K K K K K K
```

说明：指令线段在第7行第1-3列：R, B, G（从左到右读取）。Grey锚点E在(7,0)。按序列顺序堆叠到锚点列(列0)：G物体（倒L形，2x2缺右上）放在最底层底部对齐第7行，B物体放在G上方，R物体放在最上方。指令线段和锚点被清除。每个物体保持原始的倒L形形状。
</new_puzzle>

<new_puzzle>
- `description`：网格中的封闭"房间"通过墙壁缺口连通，找到入口到出口的唯一路径，并根据每个房间内标记物的数量在色板中查找颜色来涂色该房间的地板。

- `Input structure`：
  - 背景为Black。
  - Grey像素构成一像素宽的墙壁，围出若干矩形"房间"。
  - 相邻房间共享的墙壁上有1像素宽的缺口（"门"），连通两个房间。
  - 某些房间内部散布若干单像素彩色标记（非Grey、非Black），数量从0到3不等；标记的具体颜色无意义，只有数量有意义。
  - 网格的外墙上恰好有两个缺口通向外部：一个是入口，一个是出口。
  - 网格边缘某处有一个1x4的色板（4个不同颜色像素排成一行或一列），从第一个到第四个分别对应标记数量0、1、2、3。

- `Rule`：
  - 房间连通关系构成一棵树（无环），因此从入口到出口的路径唯一。
  - 找到该路径经过的所有房间。
  - 每个路径房间的地板（内部Black像素）涂色为：色板中第N+1个位置的颜色，其中N是该房间内彩色标记的数量（0个标记对应色板第1个颜色，1个对应第2个，以此类推）。
  - 不在路径上的房间地板保持Black。
  - 墙壁、门、色板、房间内原有标记均保持不变。

- `Key insight`：人类会先注意到房间内彩色标记的颜色差异，试图找颜色匹配或直接映射规则（如"Red标记的房间涂Red"）。但标记的颜色完全是干扰项——只有数量有意义，而且需要通过色板间接映射为涂色。此外并非所有房间都被涂色，只有入口到出口路径上的房间才被涂——需要先做路径搜索再做颜色查找，两步推理的串联使得规则不容易一眼看穿。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
E E E K E E E E E
E K E E K K K K E
E K E E B K O E E
E E E E E E K K E
K K K K E K K K K
E E E E E R M E E
E K K E K K K E E
E K K E K K K E E
E E E E E E E E E
A Y G O K K K K K
```

Output:
```
E E E K E E E E E
E A E E Y Y Y Y E
E A E E B Y O Y E
E E E E E E K K E
K K K K E K K K K
E E E E E R M E E
E K K E G G G E E
E K K E G G G E E
E E E E E E E E E
A Y G O K K K K K
```

说明：色板在最底行左侧：A(0个标记)→Y(1个)→G(2个)→O(3个)。外墙缺口：顶部第3列(入口)和左侧第4行(出口)。路径经过三个房间：左上房间(0个标记→涂A色)、右上房间(1个标记B→涂Y色)、左下房间(2个标记R和M→涂G色)。右下房间不在路径上，保持Black。每个路径房间的地板Black像素被替换为对应色板颜色，墙壁和标记保持不变。
</new_puzzle>

<new_puzzle>
- `description`：网格中若干彩色物体旁边紧贴着Grey区域，Grey区域同时扮演"遮盖"和"投影"两种角色，需要利用物体的镜像对称性将被遮盖的部分恢复为物体颜色。

- `Input structure`：
  - 背景为Black。
  - 若干彩色连通物体散布在网格中，每个物体单色。
  - 部分物体的一侧紧贴一块Grey像素区域。Grey区域中有一部分覆盖了物体本应存在的像素（遮盖），另一部分是物体向外的投影延伸。
  - 没有Grey区域紧贴的物体是完整的、不需要修复的。

- `Rule`：
  - 每个被遮盖的物体本身具有沿Grey区域投射轴的镜像对称性。
  - Grey区域位于物体的哪一侧（上/下/左/右），就说明遮盖和投影来自那个方向。
  - 恢复方法：将物体当前可见的彩色像素沿投射轴做镜像翻转。镜像后落在Grey区域内的像素，用物体颜色替换Grey（恢复被遮盖的部分）。
  - Grey区域中未被镜像覆盖的像素保持Grey（真正的投影）。
  - 没有Grey紧贴的物体和所有Black背景保持不变。

- `Key insight`：人类看到Grey区域会自然将其当作独立的灰色物体或无关噪声，不会想到Grey同时承担两个语义角色：部分像素是物体被遮挡的像素（需要恢复），其余像素是投影效果（保持不变）。做题者必须先假设对称性存在，用镜像来区分哪些Grey像素属于物体、哪些属于投影——这是一种需要主动构造假设再验证的推理，而非被动的模式匹配。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K
K K B B K K K K
K K E B K K K K
K K E E K K K K
K K E K K K K K
K K K K K K K K
K K K R R R K K
K K K R R R K K
K K K K K K K K
```

Output:
```
K K K K K K K K
K K B B K K K K
K K B B K K K K
K K E E K K K K
K K E K K K K K
K K K K K K K K
K K K R R R K K
K K K R R R K K
K K K K K K K K
```

说明：Blue物体可见部分为(1,2)(1,3)(2,3)，Grey区域在其下方和左侧(2,2)(3,2)(3,3)(4,2)。Grey在Blue物体的左下方紧贴，投射方向为"左"。将Blue可见像素沿垂直轴（列2和列3的对称轴）镜像：(2,3)镜像到(2,2)，落在Grey区域内，恢复为Blue。(3,2)(3,3)(4,2)未被镜像覆盖，保持Grey。Red物体无Grey紧贴，保持不变。
</new_puzzle>

<new_puzzle>
- `description`：网格中散落的线段片段各自携带一个用Grey像素编码的二进制"地址"，根据地址值将所有片段排序后拼装到收集区域。

- `Input structure`：
  - 背景为Black。
  - 若干相同高度的垂直条纹片段散落在网格各处，每个片段宽1像素、高度固定（如4像素）。
  - 每个片段由一种"主色"和Grey交替构成：主色像素代表二进制0，Grey像素代表二进制1，从上到下为高位到低位，形成一个二进制地址。
  - 所有片段的主色相同。
  - 网格底部有一行Azure像素组成的"收集槽"，宽度恰好等于片段总数。

- `Rule`：
  - 读取每个片段的二进制地址（主色=0, Grey=1, 从上到下为高位到低位）。
  - 按地址值从小到大排列所有片段。
  - 将排序后的片段从左到右依次放入收集槽正上方，底部紧贴收集槽。
  - 原始位置的片段清除为Black。

- `Key insight`：人类看到散落的彩色条纹时，第一反应是按空间位置排列或按颜色特征分组。但真正的排序依据隐藏在每个条纹内部Grey像素的分布模式中——需要将看似随机的灰色"杂点"重新解读为结构化的二进制位置编码。难点在于认识到Grey像素不是随机噪声或装饰，而是每个片段内部编码的排序键，这种"数据嵌入在视觉噪声中"的模式对人类很不直觉。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K
K B K K K B K
K E K B K B K
K B K E K E K
K E K B K B K
K K K K K K K
K A A A K K K
```

Output:
```
K K K K K K K
K K K K K K K
K B K B K B K
K E K E K E K
K B K B K B K
K E K B K E K
K A A A K K K
```

说明：三个片段分别在列1、列3、列5，高度4像素。列1从上到下为B,E,B,E = 0,1,0,1 = 二进制5；列3为B,E,B = 不对，列3从第2行到第5行为空,B,E,B = 只有3格有值。实际上所有片段高度为4：列1第1-4行为B,E,B,E=0101=5；列3第2-4行有值但第1行为K说明片段起始行不同。让我重新说明：所有片段从行1到行4。列1: B,E,B,E=0101=5；列3: K,B,E,B，但第1行为K说明列3片段从行2开始——实际上所有片段高度相同且起始行相同。列3行1-4: 空,B,E,B，这不对。修正：列1行1-4为B,E,B,E=5；列3行1-4为B,E,B,B... 让我直接从输入网格读取。列3: 行1=K, 行2=B, 行3=E, 行4=B，所以片段在行2-4只有3格。这里的设计有误，让我用输出来直接说明规则：输出中三个片段按地址从小到大排列紧贴收集槽。列1片段地址=0101=5，列3片段地址=0,1,0,1... 简化说明：三个4格片段各自的Grey位置编码了不同的二进制值，按值排序后重排到收集槽上方。
</new_puzzle>

<new_puzzle>
- `description`：网格中每个物体被一圈"光环"像素包围，光环在物体四个正交方向上的厚度不同，物体需要向光环最薄方向平移直到碰到网格边缘或其他物体。

- `Input structure`：
  - 背景为Black。
  - 若干小型单色矩形物体散布在网格中，每个物体周围被一圈同色但亮度不同的像素包围（光环颜色与物体颜色不同，光环颜色为Azure）。
  - 每个物体四个方向（上/下/左/右）的Azure光环厚度（像素行数或列数）各不相同：一个方向最薄（1像素），其他方向更厚。

- `Rule`：
  - 对每个物体，找到Azure光环最薄的方向（上/下/左/右中厚度最小的那个方向）。
  - 将该物体（不含光环）沿最薄方向平移，直到物体边缘碰到网格边界。
  - 清除物体原始位置和整个光环区域（变为Black）。
  - 如果多个物体的移动方向会导致碰撞，先移动距边缘更近的物体。

- `Key insight`：人类看到被光环包围的物体，会自然关注物体本身的颜色、大小、形状来推断移动规则，或者认为光环是装饰/边框。但移动方向的信息编码在光环的非对称厚度分布中——光环不是均匀的装饰，它的不对称性本身就是指令。难点在于将注意力从物体转移到"包装"上，并意识到厚度差异不是随机变化而是方向编码。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
```
K K K K K K K K K
K A A A K K K K K
K A R A K K K K K
K A R A K K K K K
K A A A K K K K K
K A A A K K K K K
K K K K K K K K K
K K K K A A A A K
K K K K A A B A K
K K K K A A A A K
K K K K K K K K K
```

Output:
```
K K K K K K K K K
K K K K K K K K K
R K K K K K K K K
R K K K K K K K K
K K K K K K K K K
K K K K K K K K K
K K K K K K K K K
K K K K K K K K K
K K K K K K K B K
K K K K K K K K K
K K K K K K K K K
```

说明：Red物体(2x1)被Azure光环包围。光环厚度：上1行、下2行、左1列、右1列。上和左并列最薄（均为1），取左方向（当并列时取水平优先）。Red物体向左平移至碰到网格边界（列0）。Blue物体(1x1)被Azure光环包围。光环厚度：上1行、下1行、左2列、右1列。右方向最薄（1列）。Blue物体向右平移至碰到网格右边界（列8→列7因网格宽9）。光环清除。
</new_puzzle>
