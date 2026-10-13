你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是设计全新的、机制上真正创新的 ARC 谜题。

下面给出 20 个已有 ARC 题目的描述作为参考。每个描述包含 `description`（变换目标）、`Input structure`（输入结构）、`Rule`（变换规则）、`Key insight`（核心洞见）。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="2ba387bc" lines="6-21">
<description>
Sort scattered 4×4 squares into two columns by type: frames left, solids right.

Input structure:
- bg = most common color. Multiple 4×4 objects on bg.
- Frames: colored 1px border, bg-filled 2×2 interior.
- Solids: entirely filled with one non-bg color.

Rule:
- Classify each 4×4 object as frame or solid by its interior.
- Sort each group top-to-bottom by input row.
- Stack frames in left column (cols 0–3), solids in right (cols 4–7).
- Output is 8 wide; height = 4 × max(#frames, #solids), bg-padded.

Key insight: Frame vs solid is determined solely by whether the 2×2 interior matches bg.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="981571dc" lines="6-19">
<description>
Reconstruct a 4-way symmetric mosaic by filling missing (black) pixels.

Input structure:
- A colored grid exhibiting bilateral symmetry along both a vertical and a horizontal axis.
- Some pixels are erased to color 0 (black), creating gaps in the pattern.

Rule:
- Infer the vertical and horizontal reflection axes from the visible (non-zero) pixels.
- For each black pixel, recover its color from the mirrored position across either axis or both.
- When multiple mirrors yield candidates, use the majority color.

Key insight: The reflection axes are not given — they must be deduced from the surviving pixels by finding the axes that best preserve the existing symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="ad173014" lines="97-111">
<description>
Recolor each framed interior mask with the color from the next red frame clockwise around the blue connector cycle.

Input structure
- The grid contains hollow red square frames connected pairwise by blue line components.
- The blue components connect the frames into one cycle, with every frame touching exactly two blue components.
- Each frame interior contains background plus exactly one non-background color mask.

Rule
- Sort the red frames clockwise by their centers around the cycle.
- Keep each frame's interior mask shape in place, but recolor it with the interior color from the next clockwise frame.
- Leave the red frames, blue connectors, background, and interior background holes unchanged.

Key insight
- The colored masks do not move; the blue cycle gives the order in which their colors rotate.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="dd6b8c4b" lines="9-22">
<description>
Move the closest reachable maroon pixels into the 3×3 block around the unique red center.

Input structure:
- bg = orange floor, magenta walls, scattered maroon (9) pixels.
- A unique red (2) pixel marks the center of a 3×3 destination block.

Rule:
- Maroon pixels connected to the destination through floor/maroon cells (not blocked by walls) are candidates.
- Rank candidates by distance to the destination; the closest ones fill the block in row-major order.
- Moved maroons are erased (replaced with floor color).

Key insight: Wall barriers restrict connectivity — only maroon pixels reachable via flood-fill through the floor qualify, so proximity alone isn't enough.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="31f7f899" lines="7-21">
<description>
Sort vertical bars by height and redistribute them across fixed colored slots.

Input structure:
- bg background; a horizontal base line of a single color spans the full grid width.
- Several monochrome vertical bars of distinct colors rise upward from the base line.

Rule:
- Rank bar shapes by height (shortest to tallest).
- Rank original x-positions (slots) left to right.
- Place the i-th shortest shape into the i-th leftmost slot, recoloring it to that slot's original color.
- The base line is preserved unchanged.

Key insight: Each slot keeps its x-position and color identity; only the bar silhouettes are redistributed among slots by height order.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="984d8a3e" lines="5-21">
<description>
Shift each row's dominant middle-color run left until its right edge aligns with the leftmost original right edge, moving displaced left-side pixels after the run.

Input structure:
- The grid has three colors.
- The most common color forms one contiguous horizontal run in every row.
- Pixels before those runs share one side color, and pixels after those runs share the other side color.

Rule:
- Find the minimum right edge column among the middle-color runs.
- In each row, keep the run width and the trailing right-side color count.
- Place the middle-color run so its right edge is at that minimum column, unless the run is too wide, in which case start it at column 0.
- Fill any remaining non-trailing cells with the left-side color.

Key insight:
The varying row prefixes are movable padding; the stable feature is the leftmost right boundary of the middle-color shape.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="7e576d6e" lines="159-181">
<description>
Draw the marker-colored route through the centers of all doors in the parallel wall maze.

Input structure:
- The grid has a background, parallel full-span walls, one three-pixel door
  segment in each wall, and two marker pixels sharing a color.
- Walls are either all vertical or all horizontal, and the markers are on
  opposite outer sides of the wall set.

Rule:
- Keep the scene unchanged except for the route.
- Visit the door centers in wall order from the upper/left marker side to
  the lower/right marker side.
- In each corridor, draw an orthogonal path to the cell just before the next
  wall, move along that wall to the door center, cross the center, and
  continue to the next corridor; after the last door, connect to the other
  marker.

Key insight:
- The three-pixel door segments mark the only crossing points through a set
  of parallel walls, so the output is the corridor path threaded through
  those centers.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="41ace6b5" lines="7-26">
<description>
Normalize vertical two-color bars between paired marker columns around the marker rows.

Input structure
- The background is the dominant color.
- A row of red marker pixels sits directly above a row of gray marker pixels
  in the same alternating columns.
- The columns between adjacent markers contain vertical bars made from a
  contiguous teal run above a contiguous blue run.

Rule
- Keep the marker pixels and all non-bar background cells fixed.
- Let h be the maximum teal-run height among the bar columns; paint teal in
  every bar column for h rows ending at the red marker row.
- In each bar column, move its blue run to start at the gray marker row and
  fill all lower cells in that column with maroon.

Key insight
- The marker rows are a baseline: teal is normalized into a common upper
  band, while each blue run keeps its own height below the baseline.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="b745798f" lines="18-33">
<description>
Expand each 3-pixel L marker into a same-color corner L on an empty background.

Input structure
- The grid is an odd square with one dominant background color.
- Every foreground object is a monochromatic 4-connected 3-pixel L inside a
  2x2 box.

Rule
- Clear the grid to the background.
- For each marker, map its L orientation to the corresponding outer corner
  and draw that corner L in the marker color with arm length half the grid.

Key insight
- The small marker is a miniature corner; its orientation, not its position,
  chooses which outer corner receives the expanded L.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="9473c6fb" lines="10-25">
<description>
Recolor the sparse foreground pixels in scan order along their longer spread, cycling red, teal, and grey.

Input structure:
- The dominant color is the background.
- Every foreground object is a single isolated 4-connected pixel.
- The foreground pixel set has a rectangular bbox whose row or column extent defines the scan direction.

Rule:
- Keep the background unchanged.
- If the foreground bbox is at least as tall as it is wide, order foreground pixels row-major; otherwise order them column-major.
- Recolor the ordered pixels by repeating red, teal, grey from the first pixel onward.

Key insight:
- The original foreground colors are only markers; the output color depends on each marker's ordinal position in the dominant scan direction.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="a25697e4" lines="9-23">
<description>
Dock movable tools into border-anchored frames by filling cavities.

Input structure:
- bg = most common color.
- Border-touching components cluster into anchor frames with interior bg cavities.
- Non-border 2-color components are movable tools.

Rule:
- Match each tool to a frame whose cavity size equals one of its color-region counts.
- Rotate/flip the tool so one color fills the cavity exactly; the other stays outside.
- If the docking orientation uses a reflection, swap the tool's two colors.

Key insight: Cavity size determines tool–frame pairing; reflection triggers color swap.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="6ffe8f07" lines="49-64">
<description>
Cast yellow horizontal and vertical rays from the teal rectangle until blue blockers stop them.

Input structure:
- A black-background grid containing one solid teal rectangle.
- Blue objects are blockers; other non-background objects may lie in the ray paths.
- Yellow is absent from the input and is the output paint color.

Rule:
- From every row of the teal rectangle, cast rays left and right to the grid edge.
- From every column of the teal rectangle, cast rays up and down to the grid edge.
- A ray stops before a blue pixel; all crossed non-blue, non-teal pixels become yellow.

Key insight:
The teal rectangle defines a full cross-shaped projection, while blue objects are walls and red/background cells are simply paintable space.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="87ab05b8" lines="9-24">
<description>
Erase all distractor pixels and expand the single red control pixel into the grid quadrant that contains it.

Input structure:
- The grid has one dominant background color.
- Exactly one red pixel is the control pixel.
- Other non-background pixels are distractors.

Rule:
- Clear the canvas to the background color.
- Split the grid into equal top/bottom and left/right halves.
- Fill the quadrant containing the red control pixel with red.

Key insight:
- The red control pixel selects a whole half-grid quadrant by its row half and column half; all other foreground pixels are noise.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="6a980be1" lines="46-61">
<description>
Project the framed periodic stripe pattern across the canvas and erase the unused frame cells.

Input structure
- A bg canvas with a one-cell-thick rectangular frame of one color.
- Inside the frame are two parallel full-interior marker bars of another color.
- A third color fills the region between the marker bars on periodically spaced rows or columns.

Rule
- Infer the periodic row or column mask from the third-color stripes inside the frame.
- Paint the frame color along those full rows or columns, extended to the grid boundary.
- Extend the marker bars across the full canvas and repaint the source-color stripe cells between them.

Key insight
- The outer frame is not preserved as a border; it supplies the color for the periodic stripes selected by the inner source pattern.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="f3e14006" lines="21-39">
<description>
Build the woven rectangle determined by the two marked cells on each crossed bar.

Input structure
A black-background grid contains one long horizontal bar and one long vertical bar.
Each bar has a dominant base color and exactly two same-colored marker pixels.

Rule
The horizontal marker pair gives the output column span.  From the horizontal bar
row, walk toward the farther vertical marker; rows and columns alternate every
other cell.  Even step/even offset cells use the horizontal marker color until
the nearer vertical marker is reached, then use the vertical marker color.
Odd row steps use the horizontal base on marker columns and the original
crossing color on spacer columns.  Spacer columns on even row steps use the
vertical base, except at the switch row where they use the crossing color.

Key insight
The original cross cell records which base color wins at base/base lattice
crossings; the nearer vertical marker is the color-switch boundary.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="8e301a54" lines="8-23">
<description>
Move every foreground object straight downward by a number of rows equal to its own pixel count.

Input structure:
- The grid has one dominant background color.
- The foreground is made of separated 4-connected monochromatic objects.
- Objects may move partly below the bottom edge of the grid.

Rule:
- Clear the canvas to the background.
- For each foreground object, compute its area in pixels.
- Redraw the object at the same columns and that many rows lower, clipping pixels below the grid.

Key insight:
- The downward displacement is the object's own area; color and shape only identify which pixels move together.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="1be83260" lines="41-58">
<description>
Decode the colored slot map inside the marked motif and rebuild the motif lattice with those colors.

Input structure:
- A blank background surrounds a rectangular lattice of identical motifs.
- The motif rows and columns are separated by one background line.
- One motif has colored marker pixels in its internal odd-row/odd-column slots.

Rule:
- The motif size is 2R+1 by 2C+1, where R by C is the motif lattice size.
- Read the marked motif's odd internal slots as the color map for the R by C output motifs.
- Slots that still contain the repeated motif color map to that repeated motif color.
- Rebuild the compact lattice on a canvas filled with the repeated motif color,
  painting each motif mask with its decoded color.

Key insight:
- The holes and filled odd slots inside one motif are a miniature legend for recoloring the entire repeated motif lattice.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="e39e9282" lines="73-81">
<description>
Project maroon markers into adjacent grey or pink 3x3 squares, erasing grey squares afterward.

Input structure
The grid has a dominant background, solid 3x3 grey and/or pink squares, and maroon marker pixels or segments directly touching one side of exactly one square.
Rule
Grey squares are erased; their adjacent maroon markers stay in place and stamp the nearest edge cell inside the grey square. Pink squares stay pink; their adjacent maroon markers disappear and stamp the corresponding cell on the square's middle row or middle column.
Key insight
The side touched by a marker determines the projection: grey uses the near edge, while pink uses the centerline, preserving the marker's coordinate along that side.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="e3721c99" lines="69-83">
<description>
Recolor grey shapes by matching their hole count to a color legend.

Input structure:
- bg = most common color; grey (5) sprites scattered across the grid.
- Colored legend shapes (non-bg, non-grey, at least 2×2, not touching the outer border) near grid edge,
  each with a distinct number of enclosed holes.

Rule:
- Each colored legend shape defines a mapping: hole count → color.
- For every grey sprite, count its holes (enclosed bg regions not touching the sprite's bounding-box border).
- Recolor the grey sprite to the matching legend color; erase grey distractors with unlisted hole counts.

Key insight: Shape identity is encoded by topology (number of holes), not geometry or size.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="f341894c" lines="63-78">
<description>
Reorient each two-pixel blue/pink segment so it reads pink then blue when viewed from its border marker.

Input structure
- The background is dominant, with one border-only marker color and blue/pink foreground segments.
- Every marker lies on the outer border and points inward along its row or column.
- Along exactly one inward ray from each marker is a single adjacent two-pixel blue/pink segment.

Rule
- For each marker, look inward perpendicular to the border side containing the marker.
- Rewrite the two blue/pink pixels on that ray so the nearer one is pink and the farther one is blue.
- Keep marker pixels, background pixels, and all other pixels unchanged.

Key insight
- The marker gives a local viewing direction; the segment's required order is relative to that marker, not to global row or column order.
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
