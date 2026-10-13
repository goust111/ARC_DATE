你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是设计全新的、机制上真正创新的 ARC 谜题。

下面给出 20 个已有 ARC 题目的描述作为参考。每个描述包含 `description`（变换目标）、`Input structure`（输入结构）、`Rule`（变换规则）、`Key insight`（核心洞见）。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="880c1354" lines="14-28">
<description>
Recolor the separated outer regions by rotating their colors one step clockwise around the yellow-orange divider.

Input structure:
- The grid contains a fixed yellow/orange divider structure.
- All other colors form separated 4-connected monochrome regions around that divider.

Rule:
- Leave the yellow and orange pixels unchanged.
- Sort the non-divider regions by their centroid angle around the divider centroid in screen-coordinate clockwise order.
- Each region keeps its cells and receives the color of the previous region in that cyclic order.

Key insight:
- The objects do not move; only the color labels cycle through the spatial order of the regions around the divider.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="dce56571" lines="8-23">
<description>
Straighten the single foreground mask into a centered horizontal segment of the same color and length.

Input structure:
- The grid has one dominant background color.
- All non-background pixels share one foreground color.
- The number of foreground pixels fits in one centered row of the grid.

Rule:
- Erase the original foreground mask.
- Count its pixels, keep its color, and paint that many pixels as a horizontal segment.
- Place the segment on the middle row with equal background margin on the left and right.

Key insight:
- The original foreground shape is irrelevant except for its color and area; the output is the area normalized into the grid's central horizontal line.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="4c3d4a41" lines="15-28">
<description>
Fit a grey bar into a grey frame by aligning projections with colored columns, then stack colors above.

Input structure:
- bg, a grey structural color forming two components: a large rectangular frame (slot) and a smaller bar with upward projections.
- Colored pixels inside the slot mark target columns.

Rule:
- Move the bar upward into the slot so its bottom edge sits one cell above the slot's interior bottom.
- Shift it horizontally so the bar's upward projections align column-wise with the colored pixels in the slot.
- Stack the colored pixels immediately above the bar projections in each column.

Key insight: The bar's projections act as keys that must align with the colored-column "locks" in the slot.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="332f06d7" lines="9-23">
<description>
Slide a solid square through a winding channel to reach a same-sized target square.

Input structure:
- bg (most common color). A river color forms a winding corridor.
- Two same-sized NxN solid squares on the river: mover and destination, each a distinct color.

Rule:
- The mover square travels along the river toward the destination.
- It can only pass through positions where the full NxN area is river.
- If reachable, the mover replaces the destination; otherwise it stops at the farthest reachable point.
- The mover's original position is filled with the river color.

Key insight: Chokepoints narrower than N pixels block the square, forcing it to stop short.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="d6542281" lines="72-91">
<description>
Complete each partial marker mask by stamping the matching multicolor sprite over it.

Input structure
- A dominant background contains one or more 8-connected multicolor sprites.
- Separate target masks use colors that already occur inside one of those sprites.
- A target mask may contain only one marker pixel or several same-alignment marker pixels.

Rule
- For each multicolor sprite, test every translation that aligns a target pixel with a
  same-color sprite pixel.
- Keep the translations with maximal exact overlap against the existing target pixels,
  rejecting translations that conflict with other foreground or overlap a source sprite.
- Stamp the whole sprite at those translations; if a non-marker sprite color occurs as
  a single source pixel, consume that source-only marker back to background.

Key insight
The scattered target pixels are partial copies: their same-color overlap with a sprite
determines where the full sprite must be completed.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="902510d5" lines="58-80">
<description>
Keep the largest 8-connected motif and replace the corner/control pixels with a solid corner triangle.

Input structure
- The background is the dominant color.
- Exactly one grid corner contains a foreground control pixel; it gives the
  corner from which the triangle is painted.
- There is one unique largest 8-connected monochrome foreground object, the
  motif that remains in the output; all other foreground pixels are controls.
- Every foreground component outside that motif is a singleton control pixel.
- The fill color is the unique control color that appears in at least two
  non-corner control pixels.

Rule
- Start from a blank background canvas and redraw only the largest motif.
- Count the non-corner control pixels; that count is the triangle side.
- Paint the fill color on the right-triangle region where those two distances
  from the marked corner's edges sum to less than the side length.

Key insight
- The scattered singleton controls are counted into a compact triangular
  stack at the marked corner; only the largest motif is not part of the count.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="42f83767" lines="23-39">
<description>
Expand the lower color grid by replacing each pixel with its matching top stencil.

Input structure
The input has a top legend and a lower dense color grid, separated by blank rows.
The legend contains left-to-right 2x2 color swatches and, to their right,
the same number of square neutral-color stencil masks.

Rule
Pair the swatches and stencil masks by left-to-right order. For every pixel in
the lower color grid, paste the paired stencil mask as one output block,
recoloring the mask pixels with that pixel's color and leaving mask holes as bg.

Key insight
The lower grid is not copied directly; it is an index grid whose colors select
the stencil shape from the legend, so each input pixel becomes one square block.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="d968ffd4" lines="15-28">
<description>
Extend each of two aligned rectangular marker objects toward the other, filling half of the gap across the full perpendicular span.

Input structure
- A dominant background contains exactly two solid, same-sized monochromatic rectangles.
- The rectangles share either the same row span or the same column span, with a background gap between their facing edges.

Rule
- If the rectangles are side-by-side, color the left half of the inter-object gap across every row with the left object's color, and the right half with the right object's color.
- If one rectangle is above the other, color the upper half of the gap across every column with the upper object's color, and the lower half with the lower object's color.
- When the gap length is odd, leave the center row/column as background.

Key insight
- The marker objects are endpoints of an axial corridor; their own size identifies the facing edge, while the filled corridor spans the whole grid perpendicular to that axis.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="973e499e" lines="7-22">
<description>
Expand each input pixel into a miniature same-color mask of the whole grid.

Input structure
- The input is a non-empty two-dimensional grid of foreground colors.
- Black is reserved for the empty/background pixels introduced in the expanded output.
- The output has one input-shaped block for each input pixel.

Rule
- For every input pixel with color c, replace that pixel by a block the same shape as the input.
- Inside the block, write c exactly where the original input also has color c.
- Write black everywhere else in the block.

Key insight
Each pixel selects the global mask of its own color; the result is the blockwise self-mask of the grid.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="2f767503" lines="66-81">
<description>
Erase each target-color object crossed by the ray pointed by the bar-and-control marker.

Input structure:
- The background is the dominant color.
- There is one three-pixel straight marker bar and one singleton control pixel adjacent to the bar center.
- The remaining non-background color forms multiple 4-connected target objects.

Rule:
- Cast a straight ray from the marker bar center in the direction away from the adjacent control pixel.
- Find every 4-connected target-color object that intersects that ray.
- Recolor all pixels of those hit target objects to the background; all other pixels stay unchanged.

Key insight:
- The whole object is erased when the center ray touches any of its pixels; the full marker bar width is not swept.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="c3fa4749" lines="79-96">
<description>
Recolor each side-marked multicolor object embedded in a monochrome panel to its marker color.

Input structure:
- Same-color 4-connected components form monochrome panel regions.
- The bounding box of a panel may contain 4-connected non-panel-color objects.
- A target object touches exactly one side of its panel bbox, and all of its
  pixels on that side have one marker color.

Rule:
- For every target object, fill any panel-colored holes enclosed by the object.
- Recolor the object and filled holes to the side marker color.
- Leave all other pixels unchanged.

Key insight:
- A multicolor object's single contact side identifies both which object should
  be normalized and which color should replace it.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="538b439f" lines="122-142">
<description>
Reflect the solid rectangular source objects across the full separator line and color the gap with the separator color.

Input structure:
- A dominant background contains one full-width or full-height one-pixel
  separator line in a non-background color.
- One other foreground color forms solid rectangular objects, all strictly on the same side of that separator.
- Remaining non-background pixels are scattered distractors.

Rule:
- Keep the input grid shape and unchanged pixels outside each affected corridor.
- Mirror each source object across the separator line.
- For every source object, recolor the rectangular strip between the
  original object and its reflection to the separator color, then paint the
  reflected object in the source color.

Key insight:
- The full straight line is a mirror axis; the real objects are identified
  structurally as the only solid rectangles confined to one side, while
  scattered pixels are noise that may be overwritten inside the corridor.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="25c199f5" lines="30-44">
<description>
Pack the non-background rows from three separated panels into one panel, scanning panels right-to-left.

Input structure:
- A rectangular grid contains three equal-size panels separated by two full-height divider columns.
- The background is the dominant color; divider pixels use a color that appears only in the divider columns.

Rule:
- Read panels from right to left, and within each panel read rows from top to bottom.
- Keep each row that contains any non-background pixel, discarding all-background rows.
- Place the kept rows at the bottom of a single panel-shaped output, padding above with background rows.

Key insight:
- The task is row packing: the divider columns mark independent panels, and only the occupied rows are moved.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="80214e03" lines="48-64">
<description>
Extract the color of each noisy block in a row/column lattice and mirror the coarse grid horizontally.

Input structure:
- A background surrounds a rectangular lattice of monochrome foreground blocks.
- Consecutive rows in a lattice band share the same set of visible foreground colors.
- Consecutive columns in a lattice band share the same set of visible foreground colors.
- Each row-band/column-band intersection has one foreground color, with possible background holes.

Rule:
- Group non-empty rows and columns into consecutive bands by their foreground color sets.
- Replace each lattice cell by the single foreground color inside that row/column-band intersection.
- Return the resulting coarse color grid with the column order reversed.

Key insight:
- Whole-row and whole-column color sets reveal the hidden lattice even when neighboring cells share a color or contain holes.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="4df5b0ae" lines="27-41">
<description>
Pack the colored sprites along the bottom edge from smallest to largest.

Input structure
The grid has a dominant background color and several monochrome 4-connected
foreground sprites; a full-grid one-cell frame may surround the scene.

Rule
Remove any full-grid rectangular frame, sort the remaining sprites by pixel
area from smallest to largest, and paste their bounding-box sprites left to
right with each sprite bottom-aligned.

Key insight
The original locations do not determine the final slots; each sprite's area
gives the packing order, while its shape and color are preserved.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="f0100645" lines="51-68">
<description>
Slide each inner colored object horizontally into the same-colored vertical wall until it touches an obstacle.

Input structure
- The background is the dominant color.
- The only foreground colors are two non-background colors.
- One foreground color fills the left column and the other fills the right column.
- Extra 8-connected objects of each wall color appear away from that color's wall.

Rule
- Clear the extra objects from their original positions.
- For each wall color, process its 8-connected extra objects from nearest to farthest from the matching wall.
- Slide each object horizontally toward its matching wall as far as possible without overlapping the wall or an already settled object of the same color.
- Keep the two vertical wall columns fixed.

Key insight
- The side columns act like color-specific gravity walls: same-colored objects compact sideways against their own wall while preserving each object's shape.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="c920a713" lines="87-102">
<description>
Untangle overlapping colored rectangular frames into concentric square frames ordered by their visible crossings.

Input structure
- The background contains several one-cell-thick axis-aligned rectangular frames, one color per frame.
- Frames overlap, and the visible color at an overlap records a directed crossing from the hidden frame to the visible frame.
- Each foreground color still has visible pixels on the perimeter of its full rectangular bounding box.

Rule
- Infer each color's full rectangular frame from the bounding box of its visible pixels.
- Add a directed edge A -> B when B is visible at a crossing of A and B's frame perimeters.
- Find the unique path that visits every frame color once, then render those colors as one-pixel-thick concentric square frames from outside to inside.

Key insight
- The output is not a crop of the input; the crossing graph gives a single frame order, and that order is redrawn as nested rings.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="5e6bbc0b" lines="42-56">
<description>
Compact the checkerboard's moving color toward the edge marked by the teal pixel, leaving a maroon swept trail on the marker line.

Input structure
- A blue/black checkerboard with exactly one teal marker replacing a blue pixel on a non-corner edge.
- The marker's checkerboard parity identifies the moving color (blue); the opposite parity is background (black).

Rule
- If the marker is on the left or right edge, compact moving-color pixels in every row toward that side.
- If the marker is on the top or bottom edge, compact moving-color pixels in every column toward that side.
- Preserve the teal marker at its edge cell.
- On the marker's row or column, color cells swept by shifted moving pixels maroon unless the final compacted line occupies them.

Key insight
The marker selects both the compression direction and the checkerboard color class that moves; maroon records the movement trail only along that marked line.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="4e7e0eb9" lines="31-49">
<description>
Transform each 9x9 panel by mirroring across its center guide line or filling all four block slots from its lone exemplar.

Input structure
The grid is made of 9x9 panels separated by one-cell-wide grid lines. Each
panel has four monochrome 3x3 slots at the quadrant positions, with either a
full non-background center row, a full non-background center column, or no
guide line.

Rule
A panel with a full center column is reflected left-to-right, and a panel
with a full center row is reflected top-to-bottom. A panel with no guide line
has exactly one slot color that appears once and one slot color that appears
three times; recolor all four slots to the once-occurring slot color.

Key insight
The center guide line encodes the mirror axis, while panels without a guide
line are completed by copying the structurally unique slot color.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="42f14c03" lines="75-83">
<description>
Crop the foreground object whose missing cells match marker shapes, and fill those missing cells with the marker colors.

Input structure
A dominant background surrounds several monochromatic foreground objects. Exactly one object has background gaps inside its bounding box whose component shapes are matched by separate marker objects.
Rule
Find that object, crop its bounding box, and replace each background gap component with the color of the separate marker component or components having the same normalized shape.
Key insight
The marker objects do not mark absolute positions; their shapes identify which gaps in the selected object should be colored, while objects with unmatched gaps are distractors.
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
