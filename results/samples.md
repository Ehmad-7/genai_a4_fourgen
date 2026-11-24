# Qualitative RAG samples (draft: check every item before submitting)

## Is capmatinib effective for glioblastoma?
*(retrieval failed: no relevant abstract in the top 10)*

**Reference answer:** No. Combination of capmatinib buparlisib resulted in no clear activity in patients with recurrent PTEN-deficient glioblastoma.

**Model answer:** Yes, capmatinib is effective for glioblastoma. [3]

**Relevant abstract ids (ground truth):** [31776899]
**Retrieved chunks (passage -> abstract id, relevant?):** [1] -> 18366283 (no), [2] -> 18035958 (no), [3] -> 32557339 (no), [4] -> 22413865 (no), [5] -> 23363814 (no)

**Cited passages**
- [3] (abstract 32557339) Capmatinib (Tabrecta™) is an oral, small molecule mesenchymal-epithelial transition (MET) inhibitor being developed by Novartis Oncology, under a license from Incyte Corporation, for the treatment of lung cancer. Capmatinib targets and selectively binds to MET, including the mutant variant produced by exon 14 skipping, and inhibits cancer cell growth driven by the mutant MET variant. In May 2020, oral capmatinib received its first global approval in the USA for the treatment of adults with metastatic non-small cell lung cancer (NSCLC) whose tumours have a mutation that leads to MET exon 14 ski ...

## What is F105-P?
*(the model said it does not know)*

**Reference answer:** F105-P is a protamine-antibody fusion protein designed to deliver siRNA to HIV-infected or envelope-transfected cells. In specific, it was designed with the protamine coding sequence linked to the C terminus of the heavy chain Fab fragment of an HIV-1 envelope antibody.

**Model answer:** I don't know based on the provided passages.

**Relevant abstract ids (ground truth):** [15908939]
**Retrieved chunks (passage -> abstract id, relevant?):** [1] -> 26058333 (no), [2] -> 24214964 (no), [3] -> 30258093 (no), [4] -> 23943237 (no), [5] -> 29059618 (no)

**No citation in the answer; the two top passages:**
- [1] (abstract 26058333) p.G356D in the 9th exon and the second allele c.983G>A; p.G328E in the 8th exon of ACVR1. The most common recurrent allele c.617 G>A; p.R206H is also the most common in Indian patients with FOP.
- [2] (abstract 24214964) 'Gene' section is designed to visualize the mutation information of a driver gene in five different aspects. Moreover, a 'Meta-Analysis' function is provided so researchers may identify driver genes in customer-defined samples. The novel driver genes/mutations identified hold potential for both basic research and biotech applications.

## Is tubulin acetylation involved in cell motility?
*(retrieval worked and the answer cites a relevant passage)*

**Reference answer:** Yes, induction of alpha-tubulin acetylation correlates with inhibition of cell motility, while it is involved in additional cellular processes, e.g. cell cycle progression, differentiation, intracellular trafficking, and signalling. Dynamic microtubule (MT) acetylation/deacetylation mediating cell motility and adhesion is controlled by enzymes such as HDAC6, a major cytoplasmic α-tubulin deacetylase. While its overexpression and activation is capable to enhance cell motility, HDAC6 activity can also be negatively regulated by a number of cellular inhibitors, thus decreasing the ability of cells for migration.

**Model answer:** Yes, tubulin acetylation is involved in cell motility. [1], [4]

**Relevant abstract ids (ground truth):** [20308065, 12024216, 12677000, 20940043, 22589388, 23019416, 22193721]
**Retrieved chunks (passage -> abstract id, relevant?):** [1] -> 20940043 (yes), [2] -> 26227334 (no), [3] -> 22193721 (yes), [4] -> 12024216 (yes), [5] -> 34508164 (no)

**Cited passages**
- [1] (abstract 20940043) tubulin acetylation in multiple cellular functions, ranging from cell motility, cell cycle progression or cell differentiation to intracellular trafficking and signalling. It also points out that tubulin acetylation should no longer be seen as a passive marker of microtubule stability, but as a broad regulator of microtubule functions.
- [4] (abstract 12024216) increases alpha-tubulin acetylation. In vitro, purified HDAC6 potently deacetylates alpha-tubulin in assembled microtubules. Furthermore, overexpression of HDAC6 promotes chemotactic cell movement, supporting the idea that HDAC6-mediated deacetylation regulates microtubule-dependent cell motility. Our results show that HDAC6 is the tubulin deacetylase, and provide evidence that reversible acetylation regulates important biological processes beyond histone metabolism and gene transcription.

## Has protein citrullination been implicated in rheumatoid arthritis?
*(retrieval worked and the answer cites a relevant passage)*

**Reference answer:** Yes, protein citrullination been implicated in rheumatoid arthritis.

**Model answer:** Yes, protein citrullination has been implicated in rheumatoid arthritis. [1]

**Relevant abstract ids (ground truth):** [24498912, 24782594, 25515746, 24823363, 25475141, 24763532, 24497204, 25520183, 25182207, 25116951, 24724574, 25355199]
**Retrieved chunks (passage -> abstract id, relevant?):** [1] -> 25520183 (yes), [2] -> 25182207 (yes), [3] -> 24763532 (yes), [4] -> 25520183 (yes), [5] -> 25515746 (yes)

**Cited passages**
- [1] (abstract 25520183) OBJECTIVE: Citrullination has become a hot topic within recent years due to its involvement in diseases such as rheumatoid arthritis (RA), multiple sclerosis and fibrosis. Citrullinations are the conversion of arginine to citrulline by peptidylarginine deiminase (PAD) enzymes, which affect protein properties. The aim of this review is to summarize the advances in citrullination research and further explore the potential of citrullination as a diagnostic tool as well as inhibition of PAD enzymes as a target for treatment. METHOD: We reviewed current literature with emphasis on the role of citru ...

## Does MC1R palmitoylation reduce pigmentation?
*(retrieval worked and the answer cites a relevant passage)*

**Reference answer:** No, MC1R palmitoylation leads to increased pigmentation.

**Model answer:** No, MC1R palmitoylation does not reduce pigmentation. [1]

**Relevant abstract ids (ground truth):** [28869973]
**Retrieved chunks (passage -> abstract id, relevant?):** [1] -> 28869973 (yes), [2] -> 28869973 (yes), [3] -> 16704453 (no), [4] -> 18484624 (no), [5] -> 18366057 (no)

**Cited passages**
- [1] (abstract 28869973) therapeutic benefit by activating MC1R protein palmitoylation. MC1R palmitoylation, primarily mediated by the protein-acyl transferase ZDHHC13, is essential for activating MC1R signalling, which triggers increased pigmentation, ultraviolet-B-induced G1-like cell cycle arrest and control of senescence and melanomagenesis in vitro and in vivo. Using C57BL/6J-Mc1re/eJ mice, in which endogenous MC1R is prematurely terminated, expressing Mc1r RHC variants, we show that pharmacological activation of palmitoylation rescues the defects of Mc1r RHC variants and prevents melanomagenesis. The results hig ...
