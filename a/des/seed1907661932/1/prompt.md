你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是设计全新的、机制上真正创新的 ARC 谜题。

下面给出 20 个已有 ARC 题目的描述作为参考。每个描述包含 `description`（变换目标）、`Input structure`（输入结构）、`Rule`（变换规则）、`Key insight`（核心洞见）。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="833966f4" lines="8-22">
<description>
Swap the upper two-pixel segment and the lower two-pixel segment around the fixed center pixel.

Input structure:
- The grid is a 5x1 vertical column of individual colored pixels.
- The middle pixel is a separator between two same-size two-pixel segments.

Rule:
- Keep the middle pixel in place.
- Reverse the order of the two pixels above the middle.
- Reverse the order of the two pixels below the middle.

Key insight:
- The transformation is local vertical order reversal on each side of the center, not a color-specific rule.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="50c07299" lines="8-21">
<description>
Move the diagonal foreground segment to the next longer run along the same up-right diagonal.

Input structure
The grid has one background color and one foreground color. The foreground pixels form a
contiguous one-pixel-wide segment on a single anti-diagonal.

Rule
Erase the current segment. If its columns are c0..c1 and its length is L, draw the same
color on columns c1+1..c1+L+1 along the same anti-diagonal.

Key insight
The segment advances past its current upper-right end by one whole segment position, and
the next segment is one pixel longer than the current one.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="1ae2feb7" lines="30-45">
<description>
Extend row bars periodically across the opposite side of a separator column.

Input structure:
- bg = background. A near-solid vertical separator divides the grid into two sides.
- One side contains horizontal colored bars; the other side is blank target space.
- Bars may be stacked by row and may sit at different distances from the separator.

Rule:
- Identify the separator as the column whose non-bg cells are one color.
- Treat the denser side as the bar side and collect each contiguous row run.
- Process bars from nearest to farthest from the separator; for each bar, paint its
  color every bar-length cells into the blank side without overwriting earlier paint.

Key insight: Each bar's length is the period for its extension, and nearer bars claim target cells first.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="b9e38dc0" lines="7-18">
<description>
Fill the region enclosed by a wall sprite and radiate color outward through its opening.

Input structure:
- bg = most common color. Wall = most frequent non-bg color, forming a partial enclosure (e.g. U-shape).
- A fill-color pixel sits inside the enclosed area. Other non-bg pixels may act as obstacles.
Rule:
- Flood-fill the enclosed area with the fill color, radiating outward through the wall's open end.
- Obstacles cast "shadows" along the radiate direction, blocking fill behind them.

Key insight: The wall endpoints define the opening direction; fill radiates outward through it, and obstacles block propagation along that same axis.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="337b420f" lines="41-55">
<description>
Keep the larger sprite from each separated panel and overlay those sprites into one 5x5 canvas.

Input structure:
- The input is a 5x17 grid: three 5x5 panels separated by one-cell black columns.
- Each panel has the same background color and exactly one foreground color split into two 4-connected components.

Rule:
- In each panel, discard the smaller foreground component and keep the unique largest component as the sprite.
- Process panels left to right, placing each sprite at its panel coordinates on a 5x5 background canvas.
- If a sprite would cover an already placed sprite, slide it by the unique nearest in-bounds offset that avoids overlap.

Key insight:
- The extra same-colored component in each panel is a distractor; the final answer is the non-overlapping overlay of the three largest components.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="9f8de559" lines="19-37">
<description>
Erase the first border pixel hit by extending the red marker away from its pink trail.

Input structure
- The dominant background surrounds a unique red marker and a four-pixel
  pink trail.
- The pink pixels form one straight 8-connected ray starting next to the
  red marker.
- In the opposite direction from that trail, the red marker faces a
  non-background border or obstacle pixel beyond background cells.

Rule
- Keep the whole grid fixed except for the first non-background pixel on
  the ray from the red marker away from the pink trail.
- Recolor that one hit pixel to the background color.

Key insight
- The pink trail marks where the red marker came from, so the edit is found
  by ray-casting from the red marker in the opposite direction.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="c803e39c" lines="31-40">
<description>
Expand the second mask into a block grid by stamping the first mask with the two payload colors.

Input structure
- A background grid has four panels separated by three full-height separator columns.
- The first two panels contain same-size binary masks; the last two contain solid same-size color squares.
Rule
- Use the second mask as an outer layout. For each foreground cell in that outer mask, place a tile equal to the first mask, coloring its foreground with the first solid-square color and its background with the second solid-square color; outer background cells become all second solid-square color.
Key insight
- The output is the Kronecker product of the two binary masks, with the two right-hand panel colors providing the palette.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="57edb29d" lines="88-103">
<description>
Extract the single plain rectangle and redraw it with the aligned rectangles' border-relative marker pattern.

Input structure:
- The boundary color is the background/separator.
- Foreground objects are dense rectangular regions; exactly one rectangle has only its base color.
- Other rectangles contain their base color plus marker pixels.

Rule:
- Crop the only plain rectangle; this crop is the output canvas.
- Use marked rectangles whose row span overlaps the plain rectangle's row span.
- Transfer their marker pixels by preserving each marker run's distance from the nearest rectangle edges.

Key insight:
- The marked rectangles are examples of an edge-relative pattern, so the missing plain rectangle receives the same pattern resized to its own dimensions.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="f560132c" lines="259-273">
<description>
Assemble scattered jigsaw pieces into a rectangle, recoloring each by a legend sprite's color map.

Input structure:
- bg = most common color.
- Several monochrome non-bg sprites (the "pieces"), scattered across the grid.
- One sprite is the legend: it contains a multi-color k×k patch (typically 2×2) encoding output colors.

Rule:
- Tile all pieces (rotations/reflections allowed) into a tight rectangle with no gaps.
- The legend's k×k color patch defines a spatial color map over the rectangle.
- Each piece is recolored to the legend-map color corresponding to its position in the assembled rectangle.

Key insight: The legend piece doubles as both a jigsaw piece and a color key — its internal k×k palette assigns colors by spatial quadrant.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="271d71e2" lines="9-22">
<description>
Move bordered containers toward same-colored destination markers, charging gray cells to orange.

Input structure:
- bg = 6 (magenta). Black-bordered (0) rectangles with gray (5) / orange (7) interiors.
- A head sprite (9) sits adjacent to one container edge; a matching destination sprite lies along the movement axis.

Rule:
- Each container translates toward its destination along the head-to-dest axis.
- Steps = min(head-to-dest distance, number of gray interior cells).
- Gray cells become orange from the head-facing side inward, one per step.

Key insight: Gray cell count caps movement — a container stops short if it runs out of gray cells to charge.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="963c33f8" lines="86-108">
<description>
Drop the columns of the top marker downward, stopping blue-ended columns on grey obstacles and letting maroon-ended columns pass to the bottom.

Input structure
- The background is the dominant color.
- The most numerous non-background color forms stationary grey obstacle pixels.
- The other two colors form a single 3x3 top-edge marker whose top two rows
  are maroon and whose bottom row contains maroon or blue control pixels.

Rule
- Erase the top marker.
- Treat each marker column as a vertical drop retaining its three colors.
- A column whose bottom control pixel is blue stops immediately above the
  first grey pixel below that column, or bottom-aligns if none exists.
- Contiguous maroon-ended columns pass through obstacles and bottom-align;
  if such a run has a clear down-right slide from a grey ledge, shift the
  run one column right before bottom-aligning.
- When a maroon-ended drop lands through a grey bar on the bottom edge, clear
  the rest of the marker-width gap in that bottom bar.

Key insight
- The bottom row of the top marker is a control strip: blue cells mean
  "stick to the first obstacle," while maroon cells mean "fall through."
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="52364a65" lines="10-24">
<description>
Erase the two leftmost columns of every foreground object, leaving the rest of each object in place.

Input structure:
- A grid has one dominant background color and several monochromatic 4-connected foreground objects.
- Each foreground object spans at least two columns in its bounding box.

Rule:
- Find each same-color 4-connected foreground object.
- Within each object's own bounding box, set foreground pixels in the two leftmost columns to the background color.
- Leave all other pixels unchanged.

Key insight:
- The deletion is measured from each object's local left edge, not from the grid edge or from a shared global column.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="6cbe9eb8" lines="79-94">
<description>
Rebuild the foreground rectangles as a largest-to-smallest lower-left nesting.

Input structure
- A periodic wallpaper background is overlaid by foreground colors that each form a rectangular frame or solid rectangle.
- The largest foreground rectangle is the output frame; smaller foreground rectangles are the objects to nest inside it.
- A frame may be partially hidden by a smaller rectangle, but its visible pixels still determine its bbox.

Rule
- Ignore the wallpaper and crop the result to the largest foreground rectangle's bbox size.
- Sort foreground rectangles by bbox area from largest to smallest.
- Reconstruct each frame or solid rectangle, placing every next smaller rectangle in the lower-left of the current rectangle's usable region.

Key insight
- The scattered background colors are not objects; the meaningful order is the unique descending rectangle size chain, and each step nests into the previous rectangle's lower-left interior.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="db615bd4" lines="111-130">
<description>
Thicken the sparse frame and stamp the external sparse sprites into it in positional order.

Input structure
The grid has a dominant background color, a regular lattice color, one large
sparse rectangular frame, and one or more external monochrome sprites whose
pixels also sit on a two-step lattice.

Rule
Use the largest sparse rectangular perimeter as the frame. Clear that frame
region and the external sprite pixels to background, draw a solid one-pixel
border in the frame color, solidify each sparse sprite into its contiguous
two-step mask, and place the sprites inside the frame with equal gaps.
Wide frames receive sprites left-to-right by source position; tall frames
receive them top-to-bottom by source position.

Key insight
The frame supplies ordered slots: the sprite-to-slot assignment is by visible
source order, not by numeric color or by matching individual cells.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="252143c9" lines="14-33">
<description>
Keep only the center color and redraw it as a diagonal ray to its indicated corner.

Input structure:
- The grid is an odd square with a unique dominant background color.
- The center pixel is foreground.
- Every off-center pixel with the center color lies in the same diagonal
  quadrant relative to the center.

Rule:
- Fill the whole output with the background color.
- Use the quadrant of the center-colored pixels to choose one diagonal
  direction from the center.
- Draw a continuous line of the center color from the center to the grid
  edge in that direction.

Key insight:
The color at the exact center identifies the surviving color, while the
other pixels of that color identify which corner the center should connect to.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="ba1aa698" lines="41-55">
<description>
Extrapolate the moving sprite from a horizontal sequence of shared-border frames into the next standalone frame.

Input structure
- The grid is a row of equal-width framed panels sharing their vertical border columns.
- Each panel has the same fill color and one identical interior sprite at the same local columns.
- The sprite's top row changes by a constant amount from panel to panel.

Rule
- Build one output panel with the same height, frame color, fill color, and sprite.
- Continue the sprite's diagonal motion one more panel to the right.
- Because the output is a standalone panel, advance by the full panel width, not the shared-border spacing.

Key insight
- The panels are consecutive frames of motion; the output is the next frame after the sequence.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="7b80bb43" lines="8-21">
<description>
Remove diagonal staircase noise and complete broken straight lines.

Input structure:
- bg = background; one foreground color fg forming axis-aligned lines.
- Some fg pixels form 45° diagonal chains (8-connected but 4-disconnected from lines).

Rule:
- Delete all diagonal-only fg pixels (those with no orthogonal fg neighbor, only diagonal ones).
- Fill short gaps in horizontal/vertical lines where a deleted diagonal was nearby.
- Prune remaining diagonal-adjacent leaf pixels near former diagonal sites.

Key insight: Diagonal pixels are hints marking where a straight line is broken; their span sets the maximum gap size to fill.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="95755ff2" lines="45-60">
<description>
Paint straight rays from colored side markers into the hollow diamond frame until an interior foreground pixel blocks each ray.

Input structure
- The background contains one complete one-pixel-thick diamond frame.
- Colored marker pixels outside the diamond sit on the top, bottom, left, or right side of the frame's bounding box.
- Any non-background, non-frame pixels inside the diamond are blockers that remain unchanged.

Rule
- Each outside marker emits inward perpendicular to its side of the diamond bounding box.
- Along that row or column, paint only background pixels strictly inside the diamond with the marker color.
- Stop a ray when it reaches an existing interior foreground pixel; keep the frame and all original colored pixels unchanged.

Key insight
- The diamond frame defines a strict interior, while the side markers are column or row emitters whose color fills the visible interior interval until blocked.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="230f2e48" lines="19-38">
<description>
Fold each straight red rod at its black marker, leaving the gray-anchored side fixed.

Input structure:
- The background is the dominant color.
- Each foreground object is a straight horizontal or vertical 4-connected rod.
- Each rod contains many pixels of one line color, one endpoint marker, and one
  interior control marker.

Rule:
- For every rod, keep the segment from the endpoint marker through the control
  marker unchanged.
- Move the line-color tail on the far side of the control marker to a
  perpendicular ray starting next to that marker.
- Choose the perpendicular direction with more space to the grid edge.

Key insight:
- The endpoint marker identifies the fixed side of the rod, while the control
  marker is the hinge; the tail folds into the larger open half-plane.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="8e5c0c38" lines="9-21">
<description>
Clean each color layer by enforcing vertical mirror symmetry.

Input structure:
- bg = background. Several non-bg colors, each forming a nearly-symmetric pattern about some vertical axis.
- Some pixels in each color are "noise" that breaks the symmetry.

Rule:
- For each non-bg color, find the vertical axis that best explains its pixel layout.
- Remove any pixel that lacks a same-color mirror counterpart across that axis.

Key insight: Each color has its own independent symmetry axis; the task is per-color denoising, not whole-grid symmetry.
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
