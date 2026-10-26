# CodeLLMExp: A Multi-language Dataset for Automated Vulnerability Localization and Explanation in AI-Generated Code

## Description

**CodeLLMExp** is a large-scale, multi-language dataset created to advance research in the security of AI-generated code. It contains **10,400 high-quality examples** across Python, Java, and C, focusing on the CWE Top 25 Most Dangerous Software Errors. The dataset is specifically designed to train and evaluate machine learning models on the joint tasks of **Automated Vulnerability Localization (AVL)** and **Explainable AI (XAI)**. Each record provides a vulnerable code snippet, the precise line number of the flaw, a structured explanation (root cause, impact, mitigation), and a fixed version of the code.

## Structure

The dataset is shipped as `assets/codellmexp.zip`; after unpacking, everything is
under a single top-level directory `codellmexp/`:

*   `codellmexp/CodeLLMExp.jsonl`: A single JSON Lines file containing all 10,403 records. Each line is a self-contained JSON object with all attributes and metadata. This format is ideal for programmatic access.

*   `codellmexp/source/`: Contains code snippets generated from prompts sourced from established security benchmarks (CodeLMSec, SecurityEval, etc.). The files are organized by `Language/CWE/filename.ext`.

*   `codellmexp/seed/`: Contains code snippets generated from canonical seed examples to ensure coverage of specific, under-represented CWEs. The files are also organized by `Language/CWE/filename.ext`.

*   `codellmexp/cwe_seeds/`: Canonical seeds — carefully selected original vulnerable examples. Organized by language (Python, C, Java) and then by `CWE-XXX` folders. These seeds are the starting point used for augmentation.

In this repository the archive is unpacked to `data/codellmexp/` (see `make data`
and `scripts/setup_data.sh`). Windows helper files (`*:Zone.Identifier`) are
removed from the archive, the data itself is unchanged.


## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to use, modify and redistribute it with proper citation.