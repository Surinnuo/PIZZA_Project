# 🍕Pizza Smart Ops - Pizza Smart Operations System

## 1. Problem & User 

Designed for managers of small-to-medium-sized pizzerias, Pizza Smart Ops addresses traditional operational pain points such as experience-based staffing, intuitive prep work, and imprecise inventory management. Through a data-driven approach, it empowers store managers, shift supervisors, and procurement leads to achieve scientific decision-making, cost reduction, and operational efficiency.

## 2. Data 
**Data set**: pizza_sales.csv（Historical Sales Data）

**Key Fields**:
| Field | Type | Description |
| :--- | :--- | :--- |
| **order_id** | String | Unique identifier for the order |
| **pizza_name** | String | Name of the pizza |
| **quantity** | Integer | Quantity sold |
| **order_date** | Date | Date of the order |
| **order_time** | Time | Time of the order |
| **unit_price / total_price** | Decimal | Unit price / Total amount |
| **pizza_size** | Char | Size (S/M/L) |
| **pizza_category** | Char | Category (Classic/Supreme/Chicken/Veggie) |

## 3. Methods 

1. **Data Loading & Preprocessing**: Leveraging `pandas` to load CSV datasets, standardize datetime formats, and perform feature engineering for time-based analysis.
2. **Multi-Dimensional Sales Analysis**: Analyzing sales patterns across multiple dimensions, including hour of day, day of week, month, category, and pizza size.
3. **Forecasting Model Development**: Implementing time-series analysis based on historical data, integrated with seasonal adjustment algorithms for improved accuracy.
4. **Intelligent Staffing Generation**: Calculating labor requirements per time slot based on predicted sales volume to generate optimized staffing recommendations.
5. **Prep Work Planning**: Computing raw ingredient demands and preparation task lists by applying ingredient consumption coefficients to the sales forecast.
6. **API Deployment & UI**: Deploying RESTful APIs via `Flask` with a responsive Web interface for data visualization and operational insights.

**Tech Stack**: Python 3.8+ | Flask | pandas | HTML5/JavaScript

## 4. Key Findings 

* **Peak Hours & Peak Days**: Sales volume peaks consistently during dinner hours (17:00–19:00) and weekends (Friday/Saturday). This highlights a critical need for dynamic staffing and pre-peak meal preparation.
* **Category Performance**: "Classic" and "Supreme" categories dominate the revenue stream, while "Veggie" pizzas show higher price elasticity, suggesting that targeted promotions on vegetarian options could drive higher volume.
* **Size Preference vs. Profitability**: Large (L) size pizzas contribute the highest total revenue, but Medium (M) sizes often exhibit higher order frequency. Optimizing the "Upselling" strategy from M to L could further boost the average transaction value.
* **Weather & Seasonal Impact**: Significant correlation exists between external temperature/precipitation and order volume. Rainy days see a spike in delivery-heavy "Chicken" and "Supreme" categories, requiring adjusted inventory levels for specific ingredients.
* **Top Product Pairs**: By analyzing co-occurrence frequency, the system identifies the "Golden Pairs" (Top 5 product combinations) with the strongest synergy to optimize cross-selling strategies.

## 5. How to run

```bash
# 1. Clone Repository
git clone <repository_url>
cd pizza_smart_ops

# 2.Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4.Run the Application
python app.py

# 5. Open http://localhost:5002
```

## 6. Product link / Demo

**Access URL after startup:**: **http://localhost:5002**

**Core Functional Modules:**
* **Sales Data Analytics**: Multi-dimensional analysis across hours, weekdays, months, categories, and sizes.
* **Sales Forecasting**: Predictive insights for 7-day, 14-day, and 30-day time horizons.
* **Intelligent Staffing Recommendations**: Optimized daily and weekly shift scheduling plans.
* **Prep Work Task Lists**: Automated calculation of raw ingredient requirements.
* **Inventory Alerts & Monitoring**: Real-time tracking of stock levels with automated warnings.
* **Weather Impact Analysis**: Correlation analysis between weather conditions and sales fluctuations.
* **Promotion Strategy Simulator**: ROI modeling and impact assessment for various marketing campaigns.

## 7. Limitations & next steps

 **Current Limitations**
* **Data Dependency**: The forecasting model’s accuracy is highly dependent on the volume of historical data. Accuracy may be limited with insufficient data (plans to include a custom data upload feature are underway).
* **External Factors**: While real-time weather API integration is active, predictions may vary; impact analysis for external factors is currently based on simulated data models.
* **Staffing Optimization**: Staffing rules still require manual adjustments as the level of automated rationalization is still being refined.

**Future Roadmap**
* **Market Integration**: Enhance connectivity with local market dynamics by capturing real-time environmental data (e.g., local events and festivals) to analyze their impact on store sales.
* **Advanced Scheduling**: Upgrade the intelligence of the staffing engine to generate more scientifically optimized and flexible shift schedules.
* **Automated Procurement**: Implement deeper data analytics to support an automated inventory ordering workflow for store managers.

---

