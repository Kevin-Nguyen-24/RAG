"""Test dataset for RAG evaluation.

Contains ground truth question-answer pairs for ESG financial reports.
"""
from typing import List, Dict, Any
from pathlib import Path
import os


class ESGTestDataset:
    """Test dataset for Financial ESG RAG evaluation."""
    
    @staticmethod
    def _find_pdf_by_keyword(keyword: str) -> str:
        """Find PDF filename containing keyword in data directory."""
        data_dir = Path("./data")
        if not data_dir.exists():
            return f"{keyword}.pdf"  # Fallback
        
        keyword_lower = keyword.lower()
        for pdf_file in data_dir.glob("*.pdf"):
            if keyword_lower in pdf_file.name.lower():
                return pdf_file.name
        
        return f"{keyword}.pdf"  # Fallback if not found
    
    @staticmethod
    def _get_company_pdfs() -> Dict[str, str]:
        """Get actual PDF filenames for each company."""
        return {
            "absa": ESGTestDataset._find_pdf_by_keyword("absa"),
            "sasol": ESGTestDataset._find_pdf_by_keyword("sasol"),
            "clicks": ESGTestDataset._find_pdf_by_keyword("clicks"),
            "picknpay": ESGTestDataset._find_pdf_by_keyword("picknpay"),
            "distell": ESGTestDataset._find_pdf_by_keyword("distell"),
            "tongaat": ESGTestDataset._find_pdf_by_keyword("tongaat"),
            "implats": ESGTestDataset._find_pdf_by_keyword("implats"),
        }
    
    @staticmethod
    def get_test_cases() -> List[Dict[str, Any]]:
        """
        Get test cases with ground truth answers.
        
        Returns:
            List of test case dictionaries with:
            - question: User query
            - ground_truth: Expected answer
            - expected_sources: List of relevant source files
            - company: Company name (for filtering)
            - category: Question category (metrics, targets, general, comparison)
        """
        # Get actual PDF filenames dynamically
        pdfs = ESGTestDataset._get_company_pdfs()
        
        test_cases = [
    # ===== Carbon Emissions Questions =====
    {
        "question": "What is Sasol’s 2030 greenhouse gas reduction target?",
        "ground_truth": "Sasol aims to reduce its Scope 1 and 2 greenhouse gas emissions by 30% by 2030, from a 2017 baseline, and reach net zero by 2050.",
        "expected_sources": [pdfs["sasol"]],
        "company": "sasol",
        "category": "targets"
    },
    {
        "question": "How much CO2 did Implats emit in 2023 from Scope 1 and Scope 2 sources?",
        "ground_truth": "Implats reported combined Scope 1 and Scope 2 emissions of approximately 3.8 million tonnes CO2e in 2023.",
        "expected_sources": [pdfs["implats"]],
        "company": "implats",
        "category": "metrics"
    },

    # ===== Energy & Climate Questions =====
    {
        "question": "What renewable energy initiatives does Absa plan to implement by 2030?",
        "ground_truth": "Absa aims for a 30% reduction in total energy use and a 51% carbon emission reduction by 2030 through increased solar PV deployment and efficiency measures.",
        "expected_sources": [pdfs["absa"]],
        "company": "absa",
        "category": "targets"
    },
    {
        "question": "What proportion of Clicks Group’s total electricity usage was from renewable sources in 2022?",
        "ground_truth": "Clicks sourced around 20% of its electricity from renewable sources in 2022.",
        "expected_sources": [pdfs["clicks"]],
        "company": "clicks",
        "category": "metrics"
    },

    # ===== Water & Waste Questions =====
    {
        "question": "What was Distell’s total water withdrawal for 2022, and what reduction efforts are reported?",
        "ground_truth": "Distell withdrew about 4.2 million cubic metres of water in 2022 and implemented efficiency projects to reduce water intensity to 3.8 kL per kL of product.",
        "expected_sources": [pdfs["distell"]],
        "company": "distell",
        "category": "metrics"
    },
    {
        "question": "What are Pick n Pay’s goals for waste reduction by 2025?",
        "ground_truth": "Pick n Pay plans to cut single-use plastics by 50% by 2025 and ensure that all packaging is recyclable, reusable, or compostable.",
        "expected_sources": [pdfs["picknpay"]],
        "company": "pick n pay",
        "category": "targets"
    },

    # ===== Social & Governance Questions =====
    {
        "question": "What percentage of Tongaat Hulett’s management positions were held by women in 2021?",
        "ground_truth": "In 2021, women held 28.2% of top and senior management positions at Tongaat Hulett in South Africa.",
        "expected_sources": [pdfs["tongaat"]],
        "company": "tongaat hulett",
        "category": "metrics"
    },
    {
        "question": "How does Implats integrate sustainability governance within its board structure?",
        "ground_truth": "Implats’ Social, Transformation, and Remuneration (STR) Committee oversees ESG performance, with the board holding ultimate responsibility for ESG integrity and disclosure.",
        "expected_sources": [pdfs["implats"]],
        "company": "implats",
        "category": "general"
    },

    # ===== Comparison Questions =====
    {
        "question": "Compare the carbon neutrality commitments of Sasol and Absa.",
        "ground_truth": "Sasol targets net zero emissions by 2050 with a 30% reduction by 2030, while Absa targets carbon neutrality by 2050 with a 51% reduction by 2030.",
        "expected_sources": [pdfs["sasol"], pdfs["absa"]],
        "company": None,
        "category": "comparison"
    },
    {
        "question": "Which company shows higher water usage intensity — Distell or Tongaat Hulett?",
        "ground_truth": "Distell reported 3.8 kL of water per kL of product in 2022, while Tongaat Hulett’s operations, being agriculture-based, have higher water intensity due to irrigation.",
        "expected_sources": [pdfs["distell"], pdfs["tongaat"]],
        "company": None,
        "category": "comparison"
    },

    # ===== General ESG Questions =====
    {
        "question": "Which global sustainability frameworks does Pick n Pay align its ESG reporting with?",
        "ground_truth": "Pick n Pay aligns its ESG disclosures with the GRI Standards, UN SDGs, and the JSE Sustainability Disclosure Guidance.",
        "expected_sources": [pdfs["picknpay"]],
        "company": "pick n pay",
        "category": "general"
    },
    {
        "question": "What environmental focus areas are highlighted in Tongaat Hulett’s 2021 ESG report?",
        "ground_truth": "Tongaat Hulett’s key environmental focus areas include land management, water stewardship, waste management, air quality, and climate change mitigation.",
        "expected_sources": [pdfs["tongaat"]],
        "company": "tongaat hulett",
        "category": "general"
    }
]


        
        return test_cases
    
    @staticmethod
    def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
        """
        Get test cases filtered by category.
        
        Args:
            category: Category to filter by (metrics, targets, general, comparison)
            
        Returns:
            Filtered list of test cases
        """
        all_cases = ESGTestDataset.get_test_cases()
        return [case for case in all_cases if case["category"] == category]
    
    @staticmethod
    def get_test_cases_by_company(company: str) -> List[Dict[str, Any]]:
        """
        Get test cases filtered by company.
        
        Args:
            company: Company name to filter by
            
        Returns:
            Filtered list of test cases
        """
        all_cases = ESGTestDataset.get_test_cases()
        return [case for case in all_cases if case["company"] == company]
    
    @staticmethod
    def get_categories() -> List[str]:
        """Get list of available categories."""
        return ["metrics", "targets", "general", "comparison"]
    
    @staticmethod
    def get_companies() -> List[str]:
        """Get list of available companies."""
        return ["absa", "clicks", "distell", "sasol", "pick n pay"]


# Edge cases and adversarial examples
class ESGAdversarialDataset:
    """Adversarial test cases to check robustness."""
    
    @staticmethod
    def get_test_cases() -> List[Dict[str, Any]]:
        """
        Get adversarial test cases.
        
        Returns:
            List of challenging test cases
        """
        # Get actual PDF filenames dynamically
        pdfs = ESGTestDataset._get_company_pdfs()
        
        adversarial_cases = [
            # ===== Off-topic Questions =====
            {
                "question": "What is the weather like today?",
                "ground_truth": None,
                "expected_response_type": "off_topic_rejection",
                "expected_sources": [],
                "company": None,
                "category": "off_topic"
            },
            {
                "question": "Tell me a joke about sustainability.",
                "ground_truth": None,
                "expected_response_type": "off_topic_rejection",
                "expected_sources": [],
                "company": None,
                "category": "off_topic"
            },
            
            # ===== Cross-company Contamination =====
            {
                "question": "What is Absa's carbon emissions?",
                "ground_truth": "Should only cite Absa sources, not other companies.",
                "expected_sources": [pdfs["absa"]],
                "company": "absa",
                "category": "contamination_test",
                "should_not_mention": ["sasol", "clicks", "distell", "pick n pay"]
            },
            
            # ===== Missing Data =====
            {
                "question": "What was Clicks' revenue in 2025?",
                "ground_truth": None,
                "expected_response_type": "missing_data",
                "expected_sources": [],
                "company": "clicks",
                "category": "missing_data"
            },
            
            # ===== Ambiguous Questions =====
            {
                "question": "What are the emissions?",
                "ground_truth": None,
                "expected_response_type": "clarification_needed",
                "expected_sources": [],
                "company": None,
                "category": "ambiguous"
            },
            
            # ===== Hallucination Tests =====
            {
                "question": "Does Absa have a Mars colonization sustainability program?",
                "ground_truth": None,
                "expected_response_type": "missing_data",
                "expected_sources": [],
                "company": "absa",
                "category": "hallucination_test"
            },
        ]
        
        return adversarial_cases
