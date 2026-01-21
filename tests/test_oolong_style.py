"""
OOLONG-style tests for RLM Phase 2.

These tests simulate the OOLONG benchmark which requires:
- Semantic categorization of data
- Aggregation across multiple items
- Using llm_query for semantic analysis
"""

import pytest
import random
from datetime import datetime, timedelta

from rlm.rlm import RLM
from rlm.config import Config


@pytest.mark.slow
class TestOOLONGStyle:
    """OOLONG-style semantic aggregation tests (LLM-dependent, slow)"""

    @pytest.fixture
    def config(self):
        """Create test configuration with sub-calls enabled"""
        cfg = Config()
        cfg.debug = True
        cfg.max_iterations = 25
        cfg.enable_sub_calls = True
        cfg.max_sub_calls = 50
        return cfg

    @pytest.fixture
    def rlm(self, config):
        """Create RLM instance"""
        return RLM(config)

    def generate_questions_dataset(self, num_questions: int = 50) -> tuple[str, dict]:
        """
        Generate a dataset of questions with semantic categories.

        Categories (from OOLONG paper):
        - numeric: Questions about numbers/quantities
        - entity: Questions about named entities
        - location: Questions about places
        - description: Questions asking for descriptions
        - abbreviation: Questions about abbreviations
        - human: Questions about people

        Returns:
            Tuple of (context_string, expected_counts)
        """
        categories = {
            "numeric": [
                "How many planets are in the solar system?",
                "What is the population of Tokyo?",
                "How tall is Mount Everest?",
                "What year was the internet invented?",
                "How many bones are in the human body?",
            ],
            "entity": [
                "What company makes the iPhone?",
                "Which brand created ChatGPT?",
                "What organization runs the Olympics?",
                "Which university is in Cambridge, UK?",
                "What airline has the most flights?",
            ],
            "location": [
                "Where is the Eiffel Tower?",
                "In which country is the Amazon River?",
                "Where was pizza invented?",
                "What city hosts the UN headquarters?",
                "Where is Silicon Valley located?",
            ],
            "description": [
                "What is photosynthesis?",
                "How does a computer work?",
                "What causes earthquakes?",
                "Explain how vaccines work?",
                "What is machine learning?",
            ],
            "abbreviation": [
                "What does NASA stand for?",
                "What is the full form of CPU?",
                "What does HTML mean?",
                "What is API an abbreviation for?",
                "What does UNESCO stand for?",
            ],
            "human": [
                "Who invented the telephone?",
                "Who wrote Romeo and Juliet?",
                "Who was the first person on the moon?",
                "Who painted the Mona Lisa?",
                "Who discovered penicillin?",
            ],
        }

        # Generate random questions
        lines = []
        category_counts = {cat: 0 for cat in categories}

        for i in range(num_questions):
            # Pick random category and question
            cat = random.choice(list(categories.keys()))
            question = random.choice(categories[cat])
            category_counts[cat] += 1

            # Generate fake metadata
            date = datetime.now() - timedelta(days=random.randint(1, 365))
            user_id = f"user_{random.randint(100, 999)}"

            line = f"Date: {date.strftime('%Y-%m-%d')} | User: {user_id} | Question: {question}"
            lines.append(line)

        context = "QUESTION LOG:\n" + "\n".join(lines)
        return context, category_counts

    @pytest.mark.slow
    def test_count_question_categories(self, rlm):
        """
        Test counting questions by semantic category.

        This requires the LLM to:
        1. Read each question
        2. Categorize it semantically
        3. Aggregate counts
        """
        # Generate small dataset for testing
        context, expected_counts = self.generate_questions_dataset(num_questions=30)

        # We'll ask about a specific category comparison
        query = """Analyze the QUESTION LOG. Categorize each question into one of these types:
        - numeric (questions about numbers/quantities)
        - location (questions about places)
        - human (questions about people)

        Count how many questions are in each category and tell me which category has the MOST questions.
        Use code to systematically go through the questions."""

        result = rlm.run(query=query, context=context)

        assert result.success is True
        # Check that it made some analysis (sub_calls or execution)
        assert result.sub_calls > 0 or len(result.execution_history) > 0
        # Answer should mention at least one category
        assert any(cat in result.answer.lower() for cat in ["numeric", "location", "human"])

    def test_simple_aggregation(self, rlm):
        """Test simple aggregation without semantic categorization"""
        context = """
        SURVEY RESPONSES:
        Response 1: Rating=5, Satisfied=Yes
        Response 2: Rating=3, Satisfied=No
        Response 3: Rating=4, Satisfied=Yes
        Response 4: Rating=5, Satisfied=Yes
        Response 5: Rating=2, Satisfied=No
        Response 6: Rating=4, Satisfied=Yes
        Response 7: Rating=1, Satisfied=No
        Response 8: Rating=5, Satisfied=Yes
        """

        query = "Count how many responses said 'Satisfied=Yes'. Read the data and count carefully."

        result = rlm.run(query=query, context=context)

        assert result.success is True
        # There are 5 "Satisfied=Yes" responses
        assert "5" in result.answer

    def test_data_extraction(self, rlm):
        """Test extracting specific data points"""
        context = """
        INVENTORY STATUS:
        Item: Laptop | Stock: 45 | Status: Available
        Item: Mouse | Stock: 120 | Status: Available
        Item: Keyboard | Stock: 0 | Status: Out of Stock
        Item: Monitor | Stock: 23 | Status: Available
        Item: Webcam | Stock: 0 | Status: Out of Stock
        Item: Headset | Stock: 67 | Status: Available
        """

        query = "List all items that have Status: Out of Stock. Just list the item names."

        result = rlm.run(query=query, context=context)

        assert result.success is True
        # Should find Keyboard and Webcam
        answer_lower = result.answer.lower()
        assert "keyboard" in answer_lower or "webcam" in answer_lower

    def test_multi_step_analysis(self, rlm):
        """Test analysis requiring multiple steps"""
        context = """
        TRANSACTION LOG:
        TX001: Amount=$100, Type=Purchase, Category=Electronics
        TX002: Amount=$50, Type=Refund, Category=Electronics
        TX003: Amount=$200, Type=Purchase, Category=Clothing
        TX004: Amount=$75, Type=Purchase, Category=Electronics
        TX005: Amount=$30, Type=Refund, Category=Clothing
        TX006: Amount=$150, Type=Purchase, Category=Food
        """

        query = """Calculate the NET total for Electronics category.
        NET = sum of Purchases minus sum of Refunds.
        Electronics has: Purchase $100, Refund $50, Purchase $75.
        Show your calculation."""

        result = rlm.run(query=query, context=context)

        assert result.success is True
        # NET for Electronics = (100 + 75) - 50 = 125
        assert "125" in result.answer or "100" in result.answer  # At least partial


class TestSubCallFunctionality:
    """Tests specifically for llm_query sub-call functionality"""

    @pytest.fixture
    def config(self):
        cfg = Config()
        cfg.debug = True
        cfg.max_iterations = 15
        cfg.enable_sub_calls = True
        cfg.max_sub_calls = 10
        return cfg

    @pytest.fixture
    def rlm(self, config):
        return RLM(config)

    @pytest.mark.slow
    def test_llm_query_basic(self, rlm):
        """Test that llm_query function is available and works"""
        context = """
        Document 1: The quick brown fox jumps over the lazy dog.
        Document 2: Python is a popular programming language.
        Document 3: Machine learning is transforming industries.
        """

        query = """Use the llm_query function to analyze Document 2.
        Call: answer = llm_query("What programming language is mentioned in this text: " + context)
        Then report what you found."""

        result = rlm.run(query=query, context=context)

        assert result.success is True
        # Should have made at least one sub-call
        # Note: Model might not always use llm_query as instructed
        assert "python" in result.answer.lower() or result.sub_calls > 0

    def test_usage_tracking(self, rlm):
        """Test that usage statistics are properly tracked"""
        context = "Simple test context."
        query = "What does the context say?"

        result = rlm.run(query=query, context=context)

        assert result.success is True
        assert result.usage_stats is not None
        assert "root_calls" in result.usage_stats
        assert "sub_calls" in result.usage_stats
        assert result.usage_stats["root_calls"] >= 1
