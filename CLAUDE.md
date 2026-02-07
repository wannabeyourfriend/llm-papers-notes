# CLAUDE.md - Guidelines for Adding Paper Notes

This document controls how AI assistants (like Claude) should add technical paper notes to this repository.

## 🎯 Repository Philosophy

**Concise over comprehensive**: Each paper should be summarized in a single, scannable README that readers can digest in 2-3 minutes.

**Essentials only**: Focus on core techniques, performance metrics, and official links. Avoid verbose explanations, implementation tutorials, or extensive references.

## 📝 Structure for Each Paper

Every paper must have exactly **one file**: `papers/paper-name/README.md`

### Required Sections (in order)

1. **Title & Metadata**
   ```markdown
   # Paper Title
   **arXiv**: XXXX.XXXXX | **Date**: Month Year | **Team**: Institution
   [📄 PDF](link) | [🤗 Model](link) | [💻 Code](link)
   ```

2. **One-Line Summary** (🎯)
   - Maximum 2 sentences
   - Must state: what + why it matters
   - Include key innovation

3. **Key Results** (📊)
   - Table format: | Benchmark | Score | Rank |
   - Only SOTA or notable results
   - Max 10 rows

4. **Architecture** (🏗️)
   - Code block with key specs
   - Numbers only (tables/comparisons in Results)
   - Max 8 lines

5. **Core Techniques** (🔬)
   - Numbered list (1-5 items max)
   - Format: ### N. Technique Name
   - Each technique: Problem + Solution + Results
   - One paragraph each, 2-4 sentences

6. **Training** (📈)
   - Pre-training: tokens, optimizer, schedule
   - Infrastructure: parallelism, memory
   - Bullet points only

7. **Limitations** (⚠️)
   - Numbered list
   - Max 5 items
   - Direct quotes from paper preferred

8. **Links** (🔗)
   - Paper, PDF, Code, Model weights
   - Official resources only

9. **Tags**
   - **Tags**: #topic1 #topic2 #topic3
   - 3-7 relevant tags

### What NOT to Include

❌ Detailed section-by-section notes
❌ Implementation tutorials or code snippets
❌ Extensive reference lists
❌ Comparison tables with >5 models
❌ Mathematical derivations
❌ Author lists (>3 authors max, use "et al.")
❌ Acknowledgments
❌ Future work sections (unless critical)

## 📏 Length Guidelines

| Section | Max Lines | Target |
|---------|-----------|--------|
| One-Line Summary | 2 | 1-2 |
| Key Results | 12 | 8-10 |
| Architecture | 8 | 5-7 |
| Core Techniques | 30 | 20-25 |
| Training | 8 | 5-6 |
| Limitations | 10 | 5-8 |
| Links | 6 | 4-5 |
| **Total README** | **100** | **60-80** |

## 🎨 Formatting Rules

- Use emojis for section headers (🎯, 📊, 🏗️, 🔬, 📈, ⚠️, 🔗)
- Bold key terms: **MuonClip**
- Code blocks for specs: ```text```
- Tables for benchmarks only
- No nested lists beyond 2 levels
- Horizontal rule (---) before tags

## 🔍 Paper Selection Criteria

Only add papers that meet **ALL** criteria:

1. ✅ **Novel technical contribution** (not just benchmarks)
2. ✅ **Implementation details provided** (reproducible)
3. ✅ **Paradigm shift** (fundamental advance, not incremental)
4. ✅ **Open resources** (code, models, or detailed specs)

**Do NOT add**:
- Benchmark-only papers without novel techniques
- Papers without reproducibility details
- Incremental improvements (<10% gains)
- Closed-source with no code release

## 📋 When Adding a New Paper

### Step 1: Read the Paper

- Use web reader to fetch PDF/arXiv
- Focus on: abstract, introduction, method sections
- Skip: related work, extensive experiments, appendices

### Step 2: Extract Essentials

Create notes with:
```
- What problem does it solve?
- What is the core innovation? (1-2 techniques)
- What are the key results? (top 3-5 benchmarks)
- What are the key specs? (params, tokens, training time)
- What are the links? (arXiv, code, model)
```

### Step 3: Write README

Follow the template in main README.md. Keep it concise.

### Step 4: Update Main README

Add paper entry to the Papers section:
```markdown
### [N. Paper Title](./papers/paper-name/)
**arXiv**: XXXX.XXXXX | **Date**: Month Year
**Core Techniques**: Brief bullet points
**Key Results**: Top 2-3 metrics
**Links**: [Paper](link) | [Code](link)
```

Update stats: "Papers Covered: N"

### Step 5: Commit

```bash
git add .
git commit -m "Add [paper title]"
git push
```

## 🚨 Quality Checklist

Before committing, verify:

- [ ] README ≤ 100 lines
- [ ] One-line summary ≤ 2 sentences
- [ ] Core techniques ≤ 5 items
- [ ] All links are official (not blogs/summaries)
- [ ] No implementation tutorials or code snippets
- [ ] Main README updated with new paper
- [ ] Paper count incremented

## 🎯 Tone and Style

- **Objective**: No hype language ("groundbreaking", "revolutionary")
- **Technical**: Assume reader knows ML basics
- **Concise**: Remove filler words ("very", "quite", "rather")
- **Precise**: Use specific numbers (not "large", "small")
- **Direct**: Active voice, present tense

**Good**: "MuonClip enables stable 1T+ parameter training"
**Bad**: "The authors present a very innovative method that allows for..."

## 📊 Examples

### Good One-Line Summary

"Introduces QK-Clip mechanism that caps attention logits, enabling stable training of 1T+ parameter models with zero loss spikes."

### Bad One-Line Summary

"In this work, the authors present a comprehensive framework for training large language models that addresses many of the challenges faced by researchers in the field..." (too vague, too long)

### Good Core Technique

"**1. MuonClip: Stable Training**
**Problem**: Muon optimizer causes exploding attention logits (>1000) at 1T+ scale
**Solution**: Monitor max logit per head; when exceeds threshold τ=100, rescale query/key weights
**Results**: Zero loss spikes across 15.5T tokens, no performance degradation"

### Bad Core Technique

"**MuonClip Optimizer**
The MuonClip optimizer is a novel approach that combines the token-efficient Muon algorithm with a new technique called QK-Clip. This is very important because..." (too verbose, includes filler)

## 🔧 Troubleshooting

**Problem**: Paper is too long (>50 pages)
**Solution**: Focus on abstract + method section. Skip experiments.

**Problem**: Too many benchmarks
**Solution**: Only include SOTA or author-highlighted results.

**Problem**: Techniques are complex
**Solution**: Summarize the what, not the how. Focus on outcomes.

**Problem**: No official code release
**Solution**: Only add if reproducibility details are sufficient.

## 📈 Repository Maintenance

### Monthly Tasks

- [ ] Check for new papers from top labs (OpenAI, Anthropic, DeepMind, Meta, etc.)
- [ ] Update repository stats in main README
- [ ] Verify all links still work

### Quarterly Tasks

- [ ] Review existing summaries for outdated info
- [ ] Remove/annotate papers superseded by new work
- [ ] Update focus areas based on trends

---

**Last Updated**: February 2026
**Maintained By**: [Zixuan Wang](https://github.com/wannabeyourfriend)

## Quick Reference Template

```markdown
# Paper Title

**arXiv**: XXXX.XXXXX | **Date**: Month Year | **Team**: Institution

[📄 PDF](link) | [🤗 Model](link) | [💻 Code](link)

---

## 🎯 One-Line Summary

[1-2 sentences]

## 📊 Key Results

| Benchmark | Score | Rank |
|-----------|-------|------|
| ... | ... | ... |

## 🏗️ Architecture

```
[Key specs]
```

## 🔬 Core Techniques

### 1. Technique Name

**Problem**: [Issue]
**Solution**: [Method]
**Results**: [Outcome]

## 📈 Training

**Pre-training**:
- ...

## ⚠️ Limitations

1. ...
2. ...

## 🔗 Links

- **Paper**: ...
- **Code**: ...

---
```
