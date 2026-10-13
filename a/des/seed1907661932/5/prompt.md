你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是设计全新的、机制上真正创新的 ARC 谜题。

下面给出 20 个已有 ARC 题目的描述作为参考。每个描述包含 `description`（变换目标）、`Input structure`（输入结构）、`Rule`（变换规则）、`Key insight`（核心洞见）。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="305b1341" lines="22-37">
<description>
Expand each sparse color object into a filled rectangular region using its legend-paired fill color.

Input structure
A background-color grid has a two-column legend in columns 0 and 1, starting at row 0: each row gives
a source color in the left column and that source color's fill color in the right column.  Outside the
legend, the foreground pixels are sparse dot objects whose colors are exactly the legend source colors.

Rule
Clear the legend.  For each source color, take the bbox of all of its non-legend dots, expand that bbox
by one pixel in every direction, and fill background cells inside the expanded rectangle with the paired
legend fill color while preserving every original non-legend dot.

Key insight
The legend order is also the layer order: earlier legend rows are in front, so fill rectangles are painted
from the last legend row back to the first and higher-priority fills overwrite lower-priority fills.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="18447a8d" lines="33-48">
<description>
Move each right-side colored shape into the left slot whose template color is its exact complement.

Input structure:
- The grid has two equal-width slots separated by one background column.
- Non-background rows form horizontal bands separated by all-background rows.
- Each band has a shared template color in the left slot and one colored source shape in the right slot.

Rule:
- For each band, define its destination shape as the cells in the left slot not occupied by the template color.
- Normalize every right-side source shape to right-slot coordinates and match it to the unique equal destination shape.
- Clear the original right-side source pixels and paint the matched left-slot complement with the source color.

Key insight:
The right-side objects are not placed by row order; each one completes the unique left template whose missing cells have the same shape.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="af726779" lines="17-35">
<description>
Grow an alternating pink/orange triangular trace downward from the horizontal orange seed row.

Input structure:
- The grid has a green background and one horizontal row containing all orange foreground pixels.
- The seed row may contain isolated orange pixels and adjacent orange runs.
- Pink is absent from the input and is used for the first generated row.

Rule:
- Copy the input grid.
- Two rows below the current active row, place a child pixel between each active pair
  whose middle cell in the current row is background.
- Alternate generated row colors between pink and orange, stopping when no children
  can be generated or the next row would leave the grid.

Key insight:
- Only an active-background-active triplet in the current row reproduces; contiguous
  active runs do not produce children through occupied middle cells.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="d6e50e54" lines="8-25">
<description>
Move each maroon marker toward the blue rectangle and recolor the rectangle red.

Input structure
A background canvas contains one solid blue rectangular object and several isolated
maroon marker pixels. Each marker is horizontally or vertically aligned with the
rectangle and lies outside it.

Rule
Start with a background canvas, repaint the blue rectangle red, and move each
maroon marker straight toward the rectangle along its row or column. Exactly one
marker is strictly closer to the rectangle than all others and enters the
rectangle boundary; every other marker lands in the adjacent exterior cell just
outside its aligned side.

Key insight
The single entering marker is the unique global nearest marker to the rectangle,
not the nearest marker on each of the four sides.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="9f41bd9c" lines="10-30">
<description>
Move the side-attached foreground object to the opposite wall, shear its lower rows inward, and paint the crossed floor cells.

Input structure:
- The top row is the upper background color and the bottom row is the
  floor color.
- A single non-background, non-floor object touches exactly one vertical
  grid edge and rests immediately above the first solid floor row.

Rule:
- Clear the source object to the upper background color.
- Repaint each object row on the opposite side: the first three relative
  rows keep the edge alignment, and each lower row shifts one more pixel
  inward from that side.
- On the first floor row, repaint the traversed side-to-object segment
  maroon, stopping just before the final outer bottom support pixel.

Key insight:
- The side of the source object gives the horizontal direction, while the
  row index inside the object gives the one-cell-per-row diagonal shear.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="e45ef808" lines="31-46">
<description>
Mark the shallowest and deepest columns of a solid terrain boundary with vertical colored guide lines.

Input structure:
- The first row is a solid empty separator.
- Below it, each column consists of upper fill cells followed by a contiguous suffix of terrain cells.
- The bottom row is all terrain, so every column has a well-defined first terrain row.

Rule:
- Find the column whose first terrain row is latest; it is unique and its upper fill run becomes maroon.
- Find the rightmost column whose first terrain row is earliest; its upper fill run becomes yellow.
- Preserve the empty separator, terrain cells, and all other upper fill cells.

Key insight:
- The task is a column-height summary of the terrain profile: color the deepest upper run and the highest terrain peak.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="d255d7a7" lines="99-114">
<description>
Move the patterned edge band to the opposite edge along every row or column whose band slice contains the trail color.

Input structure
- The dominant color is the background.
- The two foreground colors are a frequent trail color and a rarer marker color.
- All marker pixels lie nearest to one grid edge, and that edge contains trail-color runs that define the source-band thickness.

Rule
- Choose the source edge identified by the marker pixels.
- A row or column perpendicular to that edge moves if its source-band slice contains the trail color.
- Copy that slice unchanged to the opposite edge, erase the remaining trail cells to background, and keep marker-only lines fixed.

Key insight
- The marker color identifies the active edge, while the trail color identifies which parallel rows or columns carry an edge-band slice across the grid.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="a09f6c25" lines="33-50">
<description>
Recolor each non-singleton red object by its symmetry axis and erase isolated red pixels.

Input structure:
- The grid has one dominant background color and red 4-connected foreground objects.
- Every non-singleton red object has exactly one reflection symmetry category:
  horizontal, vertical, or diagonal.
- Single red pixels are noise.

Rule:
- Start from an all-background output of the same shape.
- Recolor horizontally symmetric red objects blue, vertically symmetric objects
  green, and diagonally symmetric objects pink.
- Leave isolated red pixels as background.

Key insight: The target color is not tied to object position or size; it is a
label for the object's reflection symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="a32d8b75" lines="262-275">
<description>
Assemble and place enlarged sprites on a canvas using legend-panel instructions.

Input structure:
- Color-6 separators divide the grid into a left legend column and a main canvas.
- Each legend panel has (top to bottom): a unit sprite, a frame template, a 5x5 bordered square with a position-marker pixel, and a 3x3 rotation clock-hand.

Rule:
- Tile the unit sprite into each active cell of the frame to build an enlarged composite.
- Rotate the composite per the clock-hand direction (up=0, right=90, down=180, left=270 degrees).
- Place it on the canvas anchored at the position encoded by the marker pixel's row/col in the bordered square's 3x3 inner grid.

Key insight: The marker pixel's position in a 3x3 grid jointly encodes vertical and horizontal anchor alignment (start/middle/end).
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="9720b24f" lines="34-46">
<description>
Delete each foreign-color object whose bounding box is contained by a larger object of another color.

Input structure
- The black pixels are the empty canvas, and every other color forms one or more 8-connected monochromatic objects.
- Some larger foreground objects act as loose U/C-like containers around smaller objects of different colors.
Rule
- Find all 8-connected monochromatic non-black objects.
- Recolor an object to black exactly when its bounding box is contained in the bounding box of a larger object of another color.
- Keep all other pixels unchanged.
Key insight
- The removable objects are identified by containment in another color's rectangular extent, not by their own color or shape.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="3bd292e8" lines="13-29">
<description>
Recolor wall-separated background regions alternately, starting with grey in the lower-left region.

Input structure
- The grid has one dominant background color and one foreground wall color.
- The wall pixels form one or more 4-connected one-cell-thick segments.
- Each wall segment separates exactly two 4-connected background regions.
Rule
- Keep all wall pixels unchanged.
- Label the 4-connected background regions and connect two labels when they
  touch opposite sides of the same wall segment.
- Color the region containing the lower-left corner grey, then flip between
  grey and green across every wall adjacency.
Key insight
- The red walls define a bipartite map of regions; the lower-left background
  region fixes which side of the alternating coloring is grey.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="1a244afd" lines="49-68">
<description>
Rotate each magenta source pixel 90 degrees counterclockwise around its matched blue pivot, recoloring the new pixel orange.

Input structure:
- The background is the dominant color.
- The only foreground colors are blue pivot pixels and magenta source pixels.
- Each magenta source can be paired one-to-one with a blue pivot in the same
  row or column.

Rule:
- Remove all magenta source pixels.
- For each matched source/pivot pair, preserve the source distance from the
  pivot but rotate its offset 90 degrees counterclockwise.
- Paint the rotated in-bounds target orange; if the target is outside the
  grid, the source simply disappears.

Key insight:
The blue pixels are rotation centers, and row/column alignment gives a
unique global matching from each magenta source to its pivot.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="b5ca7ac4" lines="7-21">
<description>
Move bordered 5×5 squares to their designated grid edges by border color.

Input structure:
- bg = most common color.
- Azure (8) and red (2) bordered 5×5 squares with colored interiors, scattered across the grid.

Rule:
- Azure squares → left edge (col 0); red squares → right edge (col W−5).
- Each square keeps its original row.
- Process azure left-to-right, red right-to-left (nearest-to-edge first).
- If a square's rows overlap an already-placed edge square, place it one square-width inward instead.

Key insight: Near-edge-first ordering plus row-overlap determines which square sits at the edge vs. the second column.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="16de56c4" lines="9-34">
<description>
Fill periodic progressions that shoot from a unique grid edge.

Input structure:
- bg = most common color.  Sparse non-bg dots of several colors.
- All repeating colors originate from the same grid edge (top, bottom,
  left, or right — the one with the most non-bg dots).
- Each repeating color has exactly 2 dots per shooting-direction line;
  their gap defines the step.
- Every non-bg dot lies on a shooting-direction line that contains a
  repeating color.

Rule:
- Determine the unique shooting edge; left/right → rows, top/bottom → columns.
- For each shooting-direction line, the repeating color has exactly 2 dots;
  step = gap between them, offset = first position mod step.
- A different-colored dot aligned to the same beat (pos % step == offset) is
  a companion: it absorbs the ray and becomes the fill color, limiting the
  fill range to the span between repeating and companion dots.
- An unaligned dot is a bystander and left untouched.
- Without a companion the repeating color fills the entire line.

Key insight: the shooting edge is identified by counting boundary dots;
the gap between the two marker dots is the period; a companion sharing the
modular class acts as a wall that stops the ray and dictates the paint color.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="195c6913" lines="37-52">
<description>
Simulate particles along a path with alternating turn direction at collisions.

Input structure:
- Top-left: 2x2 monochrome blocks define a repeating color cycle (left-to-right).
- Isolated single pixels of a cycle color: start markers adjacent to the path.
- A 2x2 monochrome block of a distinct color: end marker.

Rule:
- Each start marker launches a particle moving straight, painting cycle colors in order.
- On hitting a boundary, paint that cell with the end color and turn 90°.
- Turn direction alternates: odd collisions → left (CCW), even → right (CW).
- If the scheduled turn is blocked, the particle stops.

Key insight: The alternating left/right turn rule is the core mechanism; without it the path would be trivial.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="d749d46f" lines="32-51">
<description>
Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

Input structure
The grid has one background color and one foreground color; the foreground forms
separate solid rectangles, ordered from left to right.

Rule
Build a 10-row background canvas whose width is the packed width of the wide
versions of the rectangles. For each source rectangle, paint a top-aligned
rectangle of shape min(h,w) by max(h,w), with one background column between
consecutive rectangles. Then, starting again from the left edge, paint a
bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
again separated by one background column.

Key insight
The input rectangles are not copied in their original orientations; each object
contributes one horizontal-facing and one vertical-facing normalized rectangle,
preserving left-to-right object order.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="bd5af378" lines="25-42">
<description>
Rewrite a corner L block into two nested L borders and a diagonal trace.

Input structure
A rectangular grid has exactly two colors: one color forms a one-cell-thick
L on two adjacent outer edges, and the other color fills the remaining
rectangle.

Rule
Normalize the L to the top-left corner. Paint the outer top-left L with the
fill color, the next inner L with the original border color, and the main
diagonal of the remaining inner rectangle with the fill color. All other
cells become color 8, then the original orientation is restored.

Key insight
The corner of the input L determines the orientation; after flipping to a
canonical top-left L, the output is the same nested-layer pattern for every
example.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="37ce87bb" lines="33-47">
<description>
Compute a grey result bar by adding teal bar heights and subtracting red bar heights.

Input structure
- The grid has a dominant background and several bottom-aligned one-cell-wide vertical bars.
- Teal bars are positive operands; red bars are negative operands.
- The penultimate column is an empty result slot.

Rule
- Keep every input bar unchanged.
- Compute result height = total teal bar height - total red bar height.
- Paint that many bottom cells of the penultimate column grey.

Key insight
- The bars encode arithmetic by height, with color providing the sign.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="aa62e3f4" lines="18-35">
<description>
Replace the outlined object with the outside contour of that outline in the marker color.

Input structure:
- A dominant background surrounds one 8-connected foreground outline.
- The outline is the unique largest foreground color.
- One or more smaller interior colors are inside the outline; the marker color
  is the unique smallest non-outline color.

Rule:
- Erase all original foreground pixels to background.
- Paint every background pixel that is 4-adjacent to the outline with the
  marker color.
- Keep the grid shape unchanged.

Key insight: The colored interior marker supplies the paint color, while the
largest outline supplies only the contour whose outside edge is traced.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="6bcdb01e" lines="10-24">
<description>
Extend the edge-touching green start bar through the corridor, turning at each teal blocker until the path leaves the grid.

Input structure:
- The background is the most common color.
- Exactly two foreground colors are present: a 2-pixel 4-connected start bar and many blocker pixels.
- Exactly one pixel of the start bar lies on the grid border.

Rule:
- Start from the border pixel of the bar and move through the other bar pixel into the grid.
- Paint every traversed non-blocker pixel with the start color.
- When the next cell is a blocker, turn to the unique perpendicular non-blocker neighbor; when the next cell is outside the grid, stop.

Key insight: The blocker pixels define a one-cell-wide corridor, so every interior collision has exactly one valid 90-degree turn.
</description>
</reference_puzzle>

---

# 设计原则

## 一、什么是伪创新

以下模式你必须避免：

**示例 1**：参考题是"按面积排序矩形"，新题改为"按周长排序三角形"。→ 只换了度量和形状，核心机制相同。

**示例 2**：参考题是"缺口编码方向，种子从缺口射出"，新题改为"开口编码方向，液体从开口流出"。→ 只换了隐喻，机制一模一样。

**示例 3**：把"按面积排序"包装成"遮挡推理——从重叠逆推图层深度"。→ 规则实际上不需要遮挡推理，描述夸大了认知难度。

**判断方法**：把新题和每道参考题各用10字以内概括核心机制。如果本质相同（只替换了名词/形容词），就是伪创新。

## 二、人类认知 vs 你的认知

你对"什么对人类来说是难的"有系统性误判。

**你以为隐蔽、但人类一眼就能看出来的：**

| 你常用的机制 | 人类的实际体验 |
|---|---|
| 数缺口/孔洞/标记点数量 → 查表 | 人类本能地数数，这只是算术，不是洞见 |
| 缺口/开口方向 → 编码运动方向 | 人类自然联想到方向 |
| 对称性有一个像素不对 → 检测缺陷 | 人类对对称性破缺极其敏感 |
| 物体面积/周长 → 排序或索引 | 大小是人类最先注意的属性 |
| 箭头/L形/楔形朝向 → 方向指令 | 形状方向性是人类最直觉的感知 |
| 凸凹形状互补 → 拼图匹配 | 人类从小玩拼图，这是本能 |

你觉得这些"隐蔽"，是因为你把"需要一步推导"等同于"隐蔽"。但对人类来说，上述推导都是瞬间完成的视觉直觉。**禁止**将这些作为核心难点。

**对人类真正困难的是：**

人类的难点不在于"看不见信息"，而在于**不知道该用哪个角度去解读眼前明摆着的信息**：

1. **角色反转**：看起来像装饰/边框/分隔线的元素，实际是规则的核心操作对象。人类会下意识忽略它。
2. **非预期的关联维度**：两个物体的关联不是颜色、大小、位置等常见属性，而是某种不常被注意的关系（如包围盒是否重叠、是否共线）。
3. **隐含条件分支**：不同物体执行了不同变换，但人类一开始以为是同一个变换，需要多个样例才能意识到存在分支。
4. **全局约束 vs 局部规则**：每个物体局部看有多种合理变换，但只有一种满足全局一致性。人类倾向局部思考，难以发现全局约束。
5. **输出与输入的结构不对应**：输出不是逐对象修改输入，而是提取全局信息后重新构建全新结构。人类习惯"找对应"，当对应不存在时会卡住。

**核心原则**：难点应在于**视角转换**（想不到要那样看），而不在于**信息提取**（数一下、量一下）。

## 三、如何使用参考题

**错误做法 1（大类归纳）**：把参考题抽象成"空间变换、计数编码、路径追踪、排序重排"等大类，再从这些大类出发生成。这样会导致输出落回固定模板，产生更强的同质性。

**错误做法 2（单题变体）**：挑一道参考题做维度替换/因果反转，产出的只是那道题的换皮。

**错误做法 3（简单融合）**：将两个三个题的机制单纯糅合，没有任何创新想法。

**正确做法**：深入理解每道参考题的具体机制，然后寻找**多道参考题的机制之间的交叉地带**——任何单道题都没涉及、但从多道题的机制中可以自然衍生出的新认知路径，结合多种逻辑进行创新生成新题。

**检验标准**：如果一个候选题可以被自然概括为"参考题 X 的变体"或"参考题 A + 参考题 B 的单纯组合"，则默认淘汰。只有当新题的核心认知路径对人类来说明显不同，且无法被简化回任一参考题时，才保留。

## 四、样例覆盖性

ARC 中颜色变换类题目很常见，没问题。但必须确保做题者能从 3-5 个训练样例中推断出完整规则：

- 如果规则涉及颜色映射，判断这道题是否适合在每个样例中全覆盖所有映射。适合（颜色少、自然全出现）就放心用；不适合（颜色多、每个样例只展示部分）就减少颜色种类，或在网格中放一个**可见的颜色映射图例**（如角落色块对）。
- 颜色在 ARC 中是**无序分类标签**，做题者眼中不存在颜色之间的相邻、大小、运算关系。不要设计依赖颜色编号顺序的规则。
- 同理，如果规则有 N 个分支条件，要么样例覆盖所有分支，要么网格中提供可见的图例说明。
- 自问：只给 3-5 个样例，做题者能否确定所有可能输入的输出？不能的话，减少分支或加入图例。

---

# 你的任务

## 第一阶段：机制草案

深入理解每道参考题的具体机制后，寻找多题机制之间的交叉地带。为每个候选谜题写一个**机制草案**：
- **核心机制（10字以内）**：解题者需要发现的规则本质
- **灵感来源**：借鉴了哪几道参考题的什么元素，如何交叉产生了新认知路径
- **人类视角的难点（1句话）**：做题者最可能被什么误导，规则为什么难以发现

生成 10 个左右的草案。

## 第二阶段：自检淘汰

对每个草案检验：
1. **简化测试**：用更简单的方式重述这个机制。简化后若与某道参考题本质相同，淘汰。
2. **非平凡性测试**：做题者看 2 个样例就能猜出规则，则不够难，淘汰。
3. **可视性测试**：规则依赖的信息是否都能在网格上直接看到？需要计算不可见量的（如"包围盒面积减实际面积"），淘汰或重新设计。
4. **人类难度测试**：核心难点是否落在上面"人类一眼看穿"的表里？是的话，淘汰。

保留 5以上个通过的机制。全部淘汰则回到第一阶段重新构思。

## 第三阶段：完整描述 + 样例验证

## 输出约束（强制）

第一阶段"机制草案"和第二阶段"自检淘汰"只允许在内部思考中完成，不得出现在最终输出中。

最终输出中只允许出现若干个 `<new_puzzle> ... </new_puzzle>` 块。不要输出以下任何内容：
- 分析与思考
- 草案列表
- 自检过程
- 标题
- 分隔线
- 额外解释
- 对生成过程的说明

对每个存活的机制，严格使用以下格式输出：

<new_puzzle>
- `description`：一句话概述变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：核心洞见。做题者容易走入什么误区，真正的规则为什么难以发现。

验证样例（颜色首字母：K=Black, B=Blue, R=Red, G=Green, Y=Yellow, E=Grey, M=Magenta, O=Orange, A=Azure, W=Maroon）：

Input:
[5x5 到 8x8 的具体网格]

Output:
[对应的输出网格]

简要说明此样例如何体现规则。
</new_puzzle>

---

# 格式要求

1. 不依赖大量先验知识或复杂计算，依赖对模式、结构与空间关系的直觉推理。
2. 若涉及计算，只允许简单的计数、排序与数量比较，不允许复杂数学运算。
3. 描述必须极其详细，读完后可直接写代码实现。对特定对象（如箭头）必须明确绘制方式。
4. 使用颜色名称（Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon），不使用数字编号。
5. 网格尺寸 ≤30×30，颜色不超过 10 种。
6. 输出必须以 `<new_puzzle>` 开始，并以 `</new_puzzle>` 结束；除这些块外不得包含任何其他文本。

---

# 最终审查（不需要保存到文件）

1. **原创性**：核心认知路径与任何一道参考题不同。
2. **确定性**：同一输入只能产生唯一输出。
3. **样例一致性**：验证样例的 Output 严格符合 Rule，逐像素检查。
4. **可实现性**：所有细节已给出，可直接写代码。
5. **ARC 合规性**：网格尺寸、颜色数合理，人类能从 3-5 个样例中发现规律。
