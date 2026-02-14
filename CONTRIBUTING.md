# Contributing to CreditScope

We welcome all contributions to this project! If you notice a bug or have a feature request, please [open up an issue](https://github.com/ubco-mds-2025-labs/creditscope/issues). All contributors must abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

1. Fork the repository and clone it locally.
2. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name`.
3. Make your changes and commit with clear, descriptive messages.
4. Push to your fork and submit a pull request to the `main` branch.
5. Your pull request will be reviewed by a team member before merging.

## Development Setup

```bash
conda env create -f environment.yaml
conda activate creditscope
python src/app.py
```

## Style Guidelines

- Python code should follow PEP 8.
- Altair charts should include proper axis labels, titles, and legends.
- Use meaningful variable and function names.
