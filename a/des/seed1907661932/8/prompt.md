你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是设计全新的、机制上真正创新的 ARC 谜题。

下面给出 20 个已有 ARC 题目的描述作为参考。每个描述包含 `description`（变换目标）、`Input structure`（输入结构）、`Rule`（变换规则）、`Key insight`（核心洞见）。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="9bbf930d" lines="61-72">
<description>
Move magenta particles along corridors formed by parallel colored bars.

Input structure:
- bg=7. Particles=6 in col 0. Colored bars on even rows form corridors on odd rows.

Rule:
- Particle goes straight through bg cells.
- When blocked by a colored cell, turns 90° into the perpendicular direction
  that has a bg cell flanked by same-color bars. Stops at grid edge or full blockage.

Key insight: Same-color parallel bars define the only valid corridors, so each particle follows a deterministic local trace.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="9385bd28" lines="71-87">
<description>
Fill rectangles between diagonal marker pairs using a color legend.

Input structure:
- bg grid with a vertical legend at the bottom-left: each row is a (marker_color, fill_color) pair.
- Scattered L-shaped corner markers of each marker color define opposite corners of axis-aligned rectangles.
- Some markers are solid 2×2 blocks instead of corner pairs.

Rule:
- Read legend bottom-to-top; for each entry find all pixels of that marker color.
- Determine which two bbox corners carry L-shaped anchors — the *other* diagonal defines the fill region.
- Fill the diamond (all shortest-path cells between those two corners) with the legend's fill color.
- Solid 2×2 marker blocks are simply recolored to the fill color.
- Later legend entries are protected from earlier fills.

Key insight: The L-shaped anchors sit on the marker diagonal; the fill goes on the opposite diagonal of the bounding box.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="83eb0a57" lines="72-86">
<description>
Crop the largest object and overlay each smaller sprite where its shared marker pattern matches.

Input structure
- The background is the most common color, and the foreground has one largest destination object plus smaller source sprites.
- Each source sprite shares exactly one marker color with the destination object.

Rule
- Output the tight bbox crop of the largest object.
- Translate each source sprite so every connected marker component of the shared color exactly matches destination marker components.
- Paste source sprites in decreasing area order, so smaller sprites can sit on top of larger ones when they overlap.

Key insight
- The shared-color marker components are alignment keys; matching whole marker components distinguishes singleton slots from larger same-color marker blocks.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="3dc255db" lines="37-53">
<description>
Eject content pixels from cup-shaped frames outward along each cup's tail direction.

Input structure:
- bg = background. Hollow cup/Y/funnel frames, each a unique border color.
- Each cup has a convergent tail spike on one side and an open mouth on the opposite side.
- Content pixels (a single non-border color per cup) sit inside the cup's mouth interior.

Rule:
- For each cup, identify its tail spike and tail direction.
- Erase all content pixels from the mouth interior.
- Draw a straight line of content-colored pixels from the spike outward in the tail direction,
  one pixel per erased content pixel; cells beyond the grid edge are dropped.

Key insight: The tail spike is the unique convergent point whose outward direction is opposite
the mouth opening where the content lives.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="abbfd121" lines="56-74">
<description>
Recover the repeating background pattern hidden under the largest solid rectangle.

Input structure:
- The grid is covered by a small row/column periodic pattern.
- Several non-pattern colors form solid rectangular foreground objects that
  overwrite the pattern.
- The requested output region is the largest of those solid rectangles.

Rule:
- Find every 4-connected monochromatic object whose bounding box is a filled
  rectangle with both dimensions greater than one.
- Treat those rectangles as masks, infer the smallest repeating tile that is
  consistent with all remaining visible pixels, and repaint the largest
  rectangle's coordinates with that tile.

Key insight: The rectangle colors are occluders; the visible pixels outside
them are enough to reconstruct the original periodic background underneath.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="ac6f9922" lines="18-40">
<description>
Summarize the enclosed table slots by the single object color inside each slot, using the slot background color for empty slots.

Input structure
- The most common color forms the outer background and all enclosed slot
  backgrounds.
- The enclosed background components form a complete rectangular array of
  slots, ordered by their positions.
- Each slot contains either no other color or exactly one object color.

Rule
- Find all 4-connected components of the most common color that do not touch
  the input edge.
- Sort their bounding boxes into row and column order to define the output
  table.
- For each slot, output its unique non-background color if present; otherwise
  output the background color.

Key insight
- The large grid is an expanded table: the enclosed background components are
  the cells, and any colored object drawn inside a cell is that cell's summary
  value.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="800d221b" lines="7-22">
<description>
Recolor a path-network by adjacent seed regions.

Input structure:
- bg = most common color.
- Path color P forms a thin orthogonal network with a solid square hub (NxN).
- Seed colors appear in colored regions (non-bg, non-P) adjacent to path branches.

Rule:
- Infer P as the non-bg color whose largest component is most network-like.
- Find the largest solid k×k all-P square (the hub); exclude it from recoloring.
- Recolor each path branch to the seed-region color it touches most.
- Recolor the hub center pixel to the seed whose branches touch the hub most.

Key insight: The hub stays uncolored; branches inherit color from adjacent seed regions by adjacency count.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="34cfa167" lines="94-117">
<description>
Complete a four-corner framed rectangle from one decorated anchor rectangle and its diagonal mate.

Input structure:
- The background is the dominant color.
- Two same-color solid rectangular anchors of the same size sit on opposite
  diagonal corners, with the first anchor above-left of the second.
- The upper-left anchor has two one-cell-thick vertical control bars to its
  right and two one-cell-thick horizontal control bars below it.

Rule:
- Copy the anchor rectangle into the two missing corners of the implied
  larger rectangle.
- Repeat the right-side control bars horizontally between the left and right
  anchors, using their spacing as the period, and draw the outer top and
  bottom borders in the first right-control color.
- Repeat the below-side control bars vertically between the top and bottom
  anchors, using the same period, and draw the outer left and right borders
  in the first below-control color.

Key insight: The small control bars are the first two ticks of two periodic
frame patterns; the diagonal anchor supplies the opposite corner and the
missing two anchor copies.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="a47bf94d" lines="178-189">
<description>
Transmit colored tokens along wire networks, converting each to its complementary 3×3 pattern.

Input structure:
- bg = background. Non-bg cells form wire networks and 3×3 tokens (solid blocks or X/diagonal patterns) near wire endpoints.

Rule:
- Every solid 3×3 token is replaced in-place by a diamond (center + 4 cardinal neighbors).
- Each token near a wire endpoint sends its complement to the opposite endpoint: solid emits X, X emits diamond.

Key insight: Token shape cycles solid → X → diamond; wires route each complement to the far end.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="4c7dc4dd" lines="7-21">
<description>
Learn a transformation rule from a training pair of framed sprites, then apply it to a test sprite.

Input structure:
- Grid contains several colored square frames (hollow borders of a single color).
- Each frame's interior holds a small pattern sprite; one frame's interior is blank (all black).
- Two frames form a training pair (their interiors share the same color palette); one remaining frame is the test case.

Rule:
- Identify the training pair by matching interior color sets; the other non-blank frame is the test input.
- Infer the rule mapping source→destination (e.g. reflection, color inversion, gap filling, endpoint extraction, or checkerboard complement).
- Apply that same rule to the test sprite; output the transformed interior.

Key insight: The puzzle is an analogy — "A is to B as C is to ?" — where you must discover the visual rule from one example pair and generalize it.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="db0c5428" lines="32-51">
<description>
Expand a 3x3 block motif into a 5x5 block motif around its missing center.

Input structure:
- Teal background surrounds one 9x9 foreground motif.
- The motif is a 3x3 array of 3x3 blocks; the middle block is empty teal.
- Colored pixels immediately around the empty middle block encode the border
  of the missing center block.

Rule:
- Keep the original 9x9 motif in place.
- Copy each of the eight surrounding 3x3 blocks to the opposite outer slot
  of the implied 5x5 block grid, without rotating it.
- Fill the center block from the eight pixels around the hole; its center is
  the motif's majority foreground color.

Key insight: The visible 9x9 motif is the middle of a larger periodic
block-level pattern, and the empty center is reconstructed from its immediate
inner boundary.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="aaecdb9a" lines="10-26">
<description>
Convert scattered foreground objects into bottom-aligned count bars in a fixed five-color order.

Input structure
The grid has a dominant background color and foreground objects in the task
palette: grey, red, teal, maroon, and pink. Objects are monochromatic
8-connected components; some palette colors may be absent.

Rule
For each palette color in order grey, red, teal, maroon, pink, count its
8-connected objects. Create a background grid with five columns and height
equal to the largest count, then fill each column upward from the bottom
with its palette color for that count.

Key insight
The output does not preserve object shapes or positions; it is a histogram
of per-color 8-connected object counts, using the learned palette order.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="9b5080bb" lines="161-172">
<description>
Recolor the small defect bars on each shared-color object with the opposite panel color.

Input structure
- The grid contains two large panel colors and one shared object color.
- Each shared-color object is a mostly solid rectangle inside exactly one panel.
Rule
- Find the near-solid rectangular core of each object.
- Missing core pixels and one-cell protrusions define two-pixel bars perpendicular to the nearest core edge.
- Recolor those bars with the other panel color; leave all other pixels unchanged.
Key insight
- The colors to add are not tied to numeric ids: each object's surrounding panel identifies the opposite panel color.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="edb79dae" lines="39-53">
<description>
Stamp recolored sprites into a framed region using placeholder positions and a color legend.

Input structure:
- bg background, a rectangular single-color frame, colored placeholder cells inside the frame.
- Outside the frame: sprite shapes (one per placeholder color) and a small legend mapping each placeholder color to an output color.

Rule:
- Find the frame and its interior placeholders.
- For each placeholder color, locate the matching-color sprite outside the frame.
- Stamp each sprite at the placeholder's position inside the frame.
- Recolor every stamped sprite to the output color defined by the legend.

Key insight: The legend pairs (overlapping two-color glyphs outside the frame) dictate the final recoloring — placeholder color selects the shape, legend maps it to the output color.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="3d6c6e23" lines="10-27">
<description>
Move each vertical stream of pixels into a bottom-aligned centered pile.

Input structure
- A mostly background grid contains one or more vertical source columns of
  foreground pixels.
- Each source column has n^2 foreground pixels, and the top-to-bottom color
  sequence splits into monochrome groups of lengths 1, 3, 5, ...

Rule
- Clear the canvas to background.
- For each source column, read the foreground colors from top to bottom and
  rebuild them at the bottom below the same column as odd-width centered
  rows of lengths 1, 3, 5, ...

Key insight
- The scattered vertical pixels are falling grains; a square number of
  grains decomposes into the odd row widths of a centered pyramid.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="79cce52d" lines="29-44">
<description>
Crop off the control row and column, then cyclically roll the payload by the two marker offsets.

Input structure
- The grid is square, with a control row on top and a control column on the left.
- The control row and column share one filler color and one single-pixel marker color.
- The lower-right square is the payload and contains none of the control colors.

Rule
- Remove the top control row and left control column.
- Roll the remaining payload down by the marker's zero-based position in the left control column.
- Roll the payload right by the marker's zero-based position in the top control row.

Key insight
- The two border markers are cyclic shift amounts for the payload, not colors to preserve.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="fbf15a0b" lines="49-69">
<description>
Crop out the grid half containing the two aligned marker pixels, then erase the markers to background.

Input structure
- The grid has a dominant background color and one non-background marker color
  that appears as exactly two cells.
- The marker cells are aligned either in one row or one column and both lie
  in the same half along that direction.
- Other non-background pixels are content inside one or both halves.

Rule
- If the markers share a row, keep the left or right half whose columns
  contain them.
- If the markers share a column, keep the top or bottom half whose rows
  contain them.
- In the kept half, recolor the marker pixels to the background color.

Key insight
- The two marker pixels are not part of the picture; their alignment and
  half-position identify which half of the doubled canvas is the answer.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="9f5f939b" lines="42-57">
<description>
Mark the missing center of each plus-shaped arrangement of four two-pixel bars with yellow.

Input structure:
- The background is the dominant color.
- The single foreground color forms isolated 4-connected objects, each a horizontal or vertical two-pixel bar.
- Some bars form an implied plus sign: two horizontal bars and two vertical bars around an empty center, with a one-pixel gap between the center and each bar.

Rule:
- Keep the input unchanged except at implied plus-sign centers.
- For every background pixel with foreground bars at offsets 2-3 cells left, right, above, and below, recolor that center yellow.
- The one-pixel gaps immediately next to the center in the four cardinal directions must remain background.

Key insight:
- The yellow pixels are not extending a line; they identify the shared missing center where four separated two-pixel bars align as a plus sign.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="981add89" lines="10-29">
<description>
Extend each top-row control pixel downward as a vertical toggle ray in its own color.

Input structure:
- The background is the dominant color.
- Non-background pixels on the top row are control pixels, each identifying
  one column and one ray color.
- Foreground regions below may intersect these control columns.

Rule:
- For every top-row control pixel, process the cells below it in the same
  column down to the bottom edge.
- A cell that already has the control color is erased to the background.
- Every other cell on that downward ray is painted with the control color.
- The top-row control pixels themselves remain unchanged.

Key insight:
The top row is a set of vertical ray launchers; each ray toggles only cells
matching its own color and overwrites all other colors along that column.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="7666fa5d" lines="12-26">
<description>
Fill the parallelogram-shaped region between parallel anti-diagonal line segments with red.

Input structure:
- bg = most common color.
- One non-bg color forms short diagonal segments, each lying on a single anti-diagonal (r+c = const).
- Segments come in parallel pairs bounding a region to be filled.

Rule:
- For each main-diagonal axis (r−c = const), collect anti-diagonal segments whose span covers that axis.
- Sort those segments by their anti-diagonal index (r+c).
- Fill all bg cells lying strictly between consecutive segment pairs along that axis with red (color 2).

Key insight: The two diagonal directions decouple — segments sit on anti-diagonals, and filling sweeps along main diagonals between them, forming parallelogram regions.
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
