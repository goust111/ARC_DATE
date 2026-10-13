<new_puzzle>
- `description`：将散布在网格中的彩色像素按各自颜色编码的固定偏移量平移到新位置。
- `Input structure`：
  - 网格有一种独特的主导背景色。
  - 前景是一组稀疏的彩色像素。
  - 每种前景色都是训练好的运动颜色之一：orange、red、maroon 或 blue。

- `Rule`：
  - 将画布清为背景色。
  - orange 像素向上平移 2 格，red 像素向左平移 2 格，maroon 像素向下平移 2 格，blue 像素向右平移 1 格。
  - 保持每个移动像素的原始颜色不变。

- `Key insight`：前景颜色本身就是运动指令；位置仅决定该颜色特定位移的落点。
</new_puzzle>

<new_puzzle>
- `description`：擦除第一个被射线击中的边框像素——射线方向由 red 标记的 pink 轨迹反方向确定。
- `Input structure`：
  - 主导背景包围着一个独特的 red 标记和一个四像素的 pink 轨迹。
  - pink 像素形成一条从 red 标记旁开始的直线 8-连通射线。
  - 在轨迹反方向，red 标记面对着背景像素之后的一个非背景边框或障碍像素。

- `Rule`：
  - 保持整个网格不变，仅对一个像素操作。
  - 从 red 标记沿远离 pink 轨迹的方向发射射线。
  - 将该射线遇到的第一个非背景像素重涂为背景色。

- `Key insight`：pink 轨迹标记了 red 标记的来源方向，所以编辑目标通过反向射线投射找到。
</new_puzzle>

<new_puzzle>
- `description`：对每个前景对象，将其像素总数作为向下的垂直位移量，把整个对象平移到新位置。
- `Input structure`：
  - 网格有一种主导背景色。
  - 前景由分离的 4-连通单色对象组成。
  - 对象可能部分移出网格底部边缘。

- `Rule`：
  - 将画布清为背景色。
  - 对每个前景对象，计算其像素面积。
  - 在相同的列位置、向下偏移该面积值行数的位置重绘对象，超出网格底部的像素被裁剪。

- `Key insight`：向下位移量等于对象自身的面积；颜色和形状只是标识哪些像素一起移动。
</new_puzzle>

<new_puzzle>
- `description`：将每根 grey 条移入其相邻的 orange 矩形中，作为从接触边开始的完整行条纹或列条纹。
- `Input structure`：
  - 场景使用 black 外部空间、yellow 填充、grey 一像素宽条和实心 orange 矩形。
  - 每根 grey 条恰好 4-触及一个 orange 矩形；某些 orange 矩形没有触及的 grey 条。
  - 触及同一 orange 矩形的 grey 条都有相同的朝向。

- `Rule`：
  - 将每个原始 grey 像素重涂为 yellow。
  - 对每个被 grey 条触及的 orange 矩形，统计分配给该矩形的 grey 像素总数。
  - 水平 grey 条将该数量的像素重绘为完整的 orange 行，从接触的上边或下边行开始。
  - 垂直 grey 条将该数量的像素重绘为完整的 orange 列，从接触的左边或右边列开始。
  - 没有触及 grey 条的 orange 矩形保持不变。

- `Key insight`：grey 像素不是按形状复制的；它们的总数被转移到相邻 orange 矩形中作为边缘对齐的行或列条纹。
</new_puzzle>

<new_puzzle>
- `description`：将每个 pink 控制像素转换为一个 teal 水平条，条的左右延伸范围等于该行中控制像素两侧的非背景像素计数。
- `Input structure`：
  - 网格有主导背景色和一个或多个 pink 控制像素。
  - 每个标记行恰好有一个 pink 像素；该行中的其他非背景像素是计数器。

- `Rule`：
  - 从全背景输出网格开始。
  - 对每个 pink 像素，统计同行中严格在其左侧和严格在其右侧的非背景像素数量。
  - 在该行上从 (列 - 左侧计数) 到 (列 + 右侧计数) 的范围内绘制 teal。

- `Key insight`：前景像素不被复制；只有它们在 pink 标记每侧的数量编码了 teal 条的长度。
</new_puzzle>
