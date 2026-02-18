# XYZ Analytics - Senior QA Engineer Assessment

Welcome to the technical assessment for the Senior QA Engineer position at xyz Analytics.

## 📁 Package Contents

```
qa-assessment/
├── docs/
│   └── ASSESSMENT_INSTRUCTIONS.docx    # Main assessment instructions (START HERE)
├── src/
│   ├── api/
│   │   └── conversation_api.py         # REST API endpoints (FastAPI)
│   └── services/
│       ├── notification_service.py     # Notification/alerting service
│       └── qa_scoring_engine.py        # Automated QA scoring engine
├── tests/
│   └── test_sample_starter.py          # Sample tests to extend
├── data/
│   └── test_fixtures.json              # Test data and fixtures
└── README.md                           # This file
```

## 🚀 Getting Started

1. **Read the Assessment Instructions**: Open `docs/ASSESSMENT_INSTRUCTIONS.docx` for complete details on the assessment tasks, evaluation criteria, and submission guidelines.

2. **Review the Codebase**: Examine the Python modules in the `src/` directory. These contain the components you'll be testing and analyzing.

3. **Use the Test Fixtures**: The `data/test_fixtures.json` file contains sample data you can use for your tests.

4. **Extend the Sample Tests**: Use `tests/test_sample_starter.py` as a starting point for your test implementation.

## 🛠 Technical Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pytest pytest-asyncio fastapi pydantic httpx

# Optional: For performance testing
pip install locust
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (requires pytest-cov: pip install pytest-cov)
pip install pytest-cov
pytest tests/ -v --cov=src
```

## 📋 Assessment Overview

| Task | Weight | Description |
|------|--------|-------------|
| Task 1 | 35% | Bug Identification & Analysis |
| Task 2 | 25% | Test Strategy & Test Plan |
| Task 3 | 25% | Test Implementation |
| Task 4 | 15% | Performance & Integration Testing Approach |

## ⏰ Timeline

- **Duration**: 5 days (Wednesday evening → Monday 8:00 AM)
- **Expected Effort**: 4-6 hours
- **Follow-up Interview**: Will be scheduled after submission review

## 📝 Submission

Submit a Git Repo or a single ZIP file containing:
1. Bug Report document (Task 1)
2. Test Strategy document (Task 2)
3. Python test files (Task 3)
4. Performance Testing document (Task 4)
5. README with setup instructions

## ⚠️ Important Notes

- The codebase contains **intentional bugs** for you to discover
- Focus on **explaining your reasoning**, not just providing answers
- Be prepared to discuss your submission in the follow-up interview
- Quality over quantity - thoughtful analysis is valued over volume

## 🔒 Confidentiality

This assessment is confidential. Please do not share the contents with others or post them publicly.

---

Good luck! If you have any questions, please contact your recruiter.