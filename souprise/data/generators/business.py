"""Synthetic business data generator for fine-tuning and retrieval benchmarking.

This module generates realistic ERP/CRM data for testing and fine-tuning
LLMs in business contexts. All data is synthetic and contains no real
customer or business information.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BusinessEntry:
    """A single business data entry for retrieval or training."""
    title: str
    content: str
    tags: List[str]

    def to_dict(self) -> dict:
        """Convert entry to dictionary."""
        return {"title": self.title, "content": self.content, "tags": self.tags}

    def to_alpaca_format(
        self,
        instruction_template: str = "Summarize the following business record: {title}",
    ) -> dict:
        """Convert entry to Alpaca format for Soup fine-tuning."""
        instruction = instruction_template.format(title=self.title)
        return {
            "instruction": instruction,
            "input": self.content,
            "output": self.content
        }

    def to_retrieval_format(self) -> dict:
        """Convert entry to format suitable for HDC indexing."""
        return {
            "id": self.title,
            "text": f"{self.title}\n{self.content}",
            "metadata": {"tags": self.tags}
        }


def generate_business_data(
    n: int = 10000,
    seed: int = 42,
    categories: Optional[List[str]] = None,
    customer_count: int = 200,
    product_count: int = 200,
) -> List[BusinessEntry]:
    """Generate synthetic business data (ERP/CRM/Controlling).

    All data is synthetic and reproducible via the seed parameter.
    Designed for benchmarking retrieval and fine-tuning LLMs.

    Args:
        n: Number of entries to generate.
        seed: Random seed for reproducibility.
        categories: List of categories to include. If None, uses default mix.
        customer_count: Number of unique synthetic customers.
        product_count: Number of unique synthetic products.

    Returns:
        List of BusinessEntry objects.
    """
    rng = np.random.RandomState(seed)
    entries: List[BusinessEntry] = []

    # Generate synthetic entities
    customers = [f"Customer_{i:04d}" for i in range(customer_count)]
    products = [f"Product_{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(product_count)]
    regions = ["North", "South", "East", "West", "US", "EU", "APAC", "Global"]
    departments = ["Sales", "Procurement", "Production", "Logistics", "Finance", "IT", "HR", "R&D"]
    statuses = {
        "invoice": ["paid", "open", "overdue", "cancelled"],
        "order": ["confirmed", "in production", "shipped", "delivered", "cancelled"],
        "ticket": ["open", "in progress", "resolved", "closed"],
    }

    # Default category distribution
    if categories is None:
        categories = ["invoice", "order", "customer", "product", "kpi", "budget"]
        weights = [0.30, 0.25, 0.20, 0.10, 0.08, 0.07]
    else:
        weights = [1.0 / len(categories)] * len(categories)

    for _ in range(n):
        category = rng.choice(categories, p=weights)
        entry = _generate_entry(rng, category, customers, products, regions, departments, statuses)
        if entry:
            entries.append(entry)

    return entries


def _generate_entry(
    rng: np.random.RandomState,
    category: str,
    customers: List[str],
    products: List[str],
    regions: List[str],
    departments: List[str],
    statuses: dict,
) -> Optional[BusinessEntry]:
    """Generate a single business entry based on category."""

    if category == "invoice":
        customer = rng.choice(customers)
        amount = rng.uniform(500, 50000)
        status = rng.choice(statuses["invoice"])
        month = rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        year = rng.choice([2024, 2025])
        region = rng.choice(regions)
        dept = rng.choice(departments)
        due_date = f"{rng.randint(1, 28)} {month} {year}"

        content = (
            f"Customer: {customer}\n"
            f"Amount: ${amount:,.2f}\n"
            f"Status: {status}\n"
            f"Region: {region}\n"
            f"Department: {dept}\n"
            f"Due Date: {due_date}\n"
            f"Payment Terms: Net 30"
        )
        tags = ["invoice", status, region.lower()]
        return BusinessEntry(
            title=f"Invoice {customer} {month} {year}",
            content=content,
            tags=tags
        )

    elif category == "order":
        customer = rng.choice(customers)
        product = rng.choice(products)
        quantity = rng.randint(1, 500)
        unit_price = rng.uniform(10, 500)
        total = quantity * unit_price
        status = rng.choice(statuses["order"])

        content = (
            f"Customer: {customer}\n"
            f"Product: {product}\n"
            f"Quantity: {quantity}\n"
            f"Unit Price: ${unit_price:,.2f}\n"
            f"Total: ${total:,.2f}\n"
            f"Status: {status}"
        )
        tags = ["order", status.replace(" ", "_"), "purchase"]
        return BusinessEntry(
            title=f"Order {customer} {product}",
            content=content,
            tags=tags
        )

    elif category == "customer":
        customer = rng.choice(customers)
        revenue = rng.uniform(10000, 500000)
        segment = rng.choice(["A", "B", "C", "Enterprise"])
        region = rng.choice(regions)
        contact = rng.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"])
        open_tickets = rng.randint(0, 10)
        last_contact_day = rng.randint(1, 28)
        last_contact_month = rng.randint(1, 12)

        content = (
            f"Customer: {customer}\n"
            f"Annual Revenue: ${revenue:,.2f}\n"
            f"Segment: {segment}\n"
            f"Region: {region}\n"
            f"Contact: {contact}\n"
            f"Open Tickets: {open_tickets}\n"
            f"Last Contact: {last_contact_day}/{last_contact_month}/2025"
        )
        tags = ["customer", "crm", segment.lower(), region.lower()]
        return BusinessEntry(
            title=f"Customer Profile {customer}",
            content=content,
            tags=tags
        )

    elif category == "product":
        product = rng.choice(products)
        stock = rng.randint(0, 5000)
        price = rng.uniform(20, 800)
        margin = rng.uniform(5, 45)
        sales_30d = rng.randint(10, 2000)
        trend = rng.choice(["rising", "stable", "declining"])

        content = (
            f"Product: {product}\n"
            f"Stock: {stock} units\n"
            f"Price: ${price:,.2f}\n"
            f"Margin: {margin:.1f}%\n"
            f"30-Day Sales: {sales_30d} units\n"
            f"Trend: {trend}"
        )
        tags = ["product", "inventory", trend]
        return BusinessEntry(
            title=f"Product {product} Metrics",
            content=content,
            tags=tags
        )

    elif category == "kpi":
        dept = rng.choice(departments)
        quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
        year = rng.choice([2024, 2025])
        metric_type = rng.choice(["Revenue", "Profit", "Growth", "Satisfaction", "Efficiency"])
        value = rng.uniform(0, 100)
        target = rng.uniform(50, 150)
        status = "On Target" if value >= target * 0.9 else "Below Target"

        content = (
            f"Department: {dept}\n"
            f"Metric: {metric_type}\n"
            f"{quarter} {year}: {value:.1f}\n"
            f"Target: {target:.1f}\n"
            f"Status: {status}"
        )
        tags = ["kpi", dept.lower(), metric_type.lower()]
        return BusinessEntry(
            title=f"KPI {dept} {metric_type} {quarter} {year}",
            content=content,
            tags=tags
        )

    elif category == "budget":
        dept = rng.choice(departments)
        year = rng.choice([2024, 2025])
        allocated = rng.uniform(100000, 1000000)
        spent = rng.uniform(0, allocated * 1.2)
        remaining = max(0, allocated - spent)
        utilization = min(100, (spent / allocated) * 100) if allocated > 0 else 0

        content = (
            f"Department: {dept}\n"
            f"Year: {year}\n"
            f"Allocated: ${allocated:,.2f}\n"
            f"Spent: ${spent:,.2f}\n"
            f"Remaining: ${remaining:,.2f}\n"
            f"Utilization: {utilization:.1f}%"
        )
        tags = ["budget", dept.lower(), f"{year}"]
        return BusinessEntry(
            title=f"Budget {dept} {year}",
            content=content,
            tags=tags
        )

    return None


def generate_alpaca_training_data(
    n: int = 10000,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> List[dict]:
    """Generate training data in Alpaca format for Soup fine-tuning.

    Creates question-answer pairs based on synthetic business data.

    Args:
        n: Number of training examples to generate.
        seed: Random seed for reproducibility.
        output_path: If provided, saves to JSONL file.

    Returns:
        List of training examples in Alpaca format.
    """
    # Generate business data
    entries = generate_business_data(n=n, seed=seed)

    # Convert to Alpaca format
    training_data = []
    for entry in entries:
        # Create different types of questions based on category
        if "Invoice" in entry.title:
            # Extract customer from title (e.g., "Invoice Customer_0123 Jan 2025")
            customer = entry.title.split()[1]
            training_data.append({
                "instruction": f"What is the status of the invoice for {customer}?",
                "input": "",
                "output": f"Based on the records: {entry.content}"
            })
            training_data.append({
                "instruction": f"Summarize the invoice for {customer}.",
                "input": "",
                "output": f"Invoice for {customer}: " + ". ".join(entry.content.split("\n")[:3])
            })

        elif "Customer Profile" in entry.title:
            customer = entry.title.split()[2]
            training_data.append({
                "instruction": f"What is the annual revenue for {customer}?",
                "input": "",
                "output": f"{customer} has an annual revenue of " +
                    next((line.split(": ")[1] for line in entry.content.split("\n")
                         if line.startswith("Annual Revenue")), "N/A")
            })

        elif "Product" in entry.title:
            product = entry.title.split()[1]
            training_data.append({
                "instruction": f"What is the current stock for {product}?",
                "input": "",
                "output": f"{product} has " +
                    next((line.split(": ")[1] for line in entry.content.split("\n")
                         if line.startswith("Stock")), "N/A") + " in stock."
            })

        elif "KPI" in entry.title:
            training_data.append({
                "instruction": f"What is the {entry.title}?",
                "input": "",
                "output": entry.content
            })

        elif "Budget" in entry.title:
            training_data.append({
                "instruction": f"What is the budget utilization for {entry.title}?",
                "input": "",
                "output": f"The utilization for {entry.title.split()[1]} is " +
                    next((line.split(": ")[1] for line in entry.content.split("\n")
                         if line.startswith("Utilization")), "N/A")
            })

    # Save to file if requested
    if output_path:
        with open(output_path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

    return training_data


if __name__ == "__main__":
    # Example usage: Generate 1000 training examples
    print("Generating synthetic business data for Souprise...")
    training_data = generate_alpaca_training_data(
        n=1000, seed=42, output_path="data/samples/business_1k.jsonl"
    )
    print(f"Generated {len(training_data)} training examples -> data/samples/business_1k.jsonl")
