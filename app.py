#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
披萨店智能运营系统 (Pizza Smart Ops)
Pizza Store Intelligent Operations System

基于历史销售数据的智能运营决策工具，提供销售预测、
智能排班、备餐规划、库存管理和促销策略等功能。

"""

import os
import json
import warnings
from datetime import datetime, timedelta
from functools import wraps

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, Response

warnings.filterwarnings('ignore')

# ============================================================================
# 应用配置
# ============================================================================

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False

# 数据路径配置
DATA_PATH = 'workspace/user_input_files/pizza_sales.csv'
DATA_DIR = 'workspace/data'
os.makedirs(DATA_DIR, exist_ok=True)

# 缓存配置
cache = {
    'sales_data': None,
    'last_update': None
}

# ============================================================================
# 核心数据模型
# ============================================================================

class SalesDataModel:
    """销售数据模型"""

    def __init__(self, df):
        self.df = df
        self._preprocess()

    def _preprocess(self):
        """数据预处理"""
        self.df['order_date'] = pd.to_datetime(
            self.df['order_date'], format='mixed'
        )
        self.df['order_time'] = pd.to_datetime(
            self.df['order_time'], format='%H:%M:%S'
        ).dt.time
        self.df['hour'] = pd.to_datetime(
            self.df['order_time'], format='%H:%M:%S'
        ).dt.hour
        self.df['day_of_week'] = self.df['order_date'].dt.dayofweek
        self.df['day_name'] = self.df['order_date'].dt.day_name()
        self.df['month'] = self.df['order_date'].dt.month
        self.df['quarter'] = self.df['order_date'].dt.quarter
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['week_of_year'] = self.df['order_date'].dt.isocalendar().week

    @property
    def date_range(self):
        """数据日期范围"""
        return self.df['order_date'].min(), self.df['order_date'].max()

    @property
    def total_sales(self):
        """总销售额"""
        return self.df['total_price'].sum()

    @property
    def total_orders(self):
        """总订单数"""
        return self.df['order_id'].nunique()

    @property
    def total_pizzas(self):
        """总披萨销量"""
        return self.df['quantity'].sum()


class IngredientModel:
    """食材消耗模型"""

    # 食材消耗系数 (kg per pizza)
    CONSUMPTION_RATES = {
        'Mozzarella Cheese': 0.15,
        'Pepperoni': 0.08,
        'Mushrooms': 0.05,
        'Red Onions': 0.04,
        'Green Olives': 0.03,
        'Tomatoes': 0.06,
        'Chicken': 0.10,
        'Bacon': 0.06,
        'Jalapeno Peppers': 0.02,
        'Garlic': 0.01,
        'Spinach': 0.04,
        'Artichokes': 0.05,
        'Feta Cheese': 0.04,
        'Goat Cheese': 0.03,
        'Anchovies': 0.02,
    }

    # 面团重量 (kg per pizza)
    DOUGH_WEIGHTS = {'S': 0.2, 'M': 0.3, 'L': 0.4}


class StaffingModel:
    """排班模型"""

    # 产能配置 (pizzas per person per hour)
    CAPACITY_PER_PERSON = 15

    # 时段配置
    TIME_SLOTS = {
        'lunch': {'hours': [11, 12, 13], 'base_staff': 3},
        'afternoon': {'hours': [14, 15, 16], 'base_staff': 2},
        'dinner': {'hours': [17, 18, 19, 20], 'base_staff': 4},
        'late_night': {'hours': [21, 22], 'base_staff': 2}
    }

    # 周末加成系数
    WEEKEND_FACTOR = 1.3

    # 高峰时段额外人手
    PEAK_HOURS = [12, 18, 19]
    PEAK_BONUS = 1


class WeatherImpactModel:
    """天气影响模型"""

    # 天气对销量的影响系数
    IMPACT_COEFFICIENTS = {
        'sunny': 1.0,
        'cloudy': 0.95,
        'rainy': 1.15,
        'heavy_rain': 1.30,
        'snowy': 1.45,
        'hot': 0.85,
        'cold': 1.10
    }

    # 价格敏感度
    PRICE_ELASTICITY = {
        'Classic': -1.2,
        'Supreme': -1.8,
        'Chicken': -1.5,
        'Veggie': -1.6
    }


class PromotionModel:
    """促销模型"""

    # 促销效果预测
    PROMOTION_EFFECTS = {
        'discount_20': {'sales_lift': 0.35, 'margin_impact': -0.15},
        'buy_one_get_one': {'sales_lift': 0.50, 'margin_impact': -0.20},
        'spend_100_save_20': {'sales_lift': 0.25, 'margin_impact': -0.10},
        'combo_meal': {'sales_lift': 0.30, 'margin_impact': -0.08},
        'new_product': {'sales_lift': 0.20, 'margin_impact': -0.05}
    }

    # 节假日系数
    HOLIDAY_FACTORS = {
        'new_year': 2.0,
        'valentine': 1.8,
        'sports_event': 2.5,
        'children_day': 1.5,
        'qixi': 2.0,
        'national_day': 1.8,
        'singles_day': 3.0,
        'christmas': 2.2
    }


# ============================================================================
# 数据加载与缓存
# ============================================================================

def load_sales_data():
    """加载销售数据"""
    if cache['sales_data'] is not None:
        return cache['sales_data']

    try:
        df = pd.read_csv(DATA_PATH)
        cache['sales_data'] = SalesDataModel(df)
        cache['last_update'] = datetime.now()
        return cache['sales_data']
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def get_data_summary():
    """获取数据摘要"""
    model = load_sales_data()
    if model is None:
        return None

    start_date, end_date = model.date_range

    return {
        'data_range': {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'days': (end_date - start_date).days + 1
        },
        'total_sales': round(model.total_sales, 2),
        'total_orders': model.total_orders,
        'total_pizzas': int(model.total_pizzas),
        'avg_order_value': round(model.total_sales / model.total_orders, 2),
        'avg_daily_sales': round(model.total_sales / ((end_date - start_date).days + 1), 2),
        'last_update': cache['last_update'].strftime('%Y-%m-%d %H:%M:%S') if cache['last_update'] else None
    }


# ============================================================================
# 分析服务层
# ============================================================================

class SalesAnalysisService:
    """销售分析服务"""

    def __init__(self, data_model):
        self.model = data_model
        self.df = data_model.df

    def get_hourly_pattern(self):
        """获取小时销量模式"""
        hourly = self.df.groupby('hour').agg({
            'quantity': ['sum', 'mean'],
            'order_id': 'nunique'
        }).reset_index()
        hourly.columns = ['hour', 'total_quantity', 'avg_quantity', 'order_count']
        return hourly.to_dict('records')

    def get_weekday_pattern(self):
        """获取星期销量模式"""
        weekday = self.df.groupby(['day_of_week', 'day_name']).agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        weekday.columns = ['day_of_week', 'day_name', 'total_quantity', 'total_revenue', 'order_count']
        weekday['avg_order_value'] = round(weekday['total_revenue'] / weekday['order_count'], 2)
        return weekday.to_dict('records')

    def get_monthly_trend(self):
        """获取月度趋势"""
        monthly = self.df.groupby(['year', 'month']).agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        monthly.columns = ['year', 'month', 'quantity', 'revenue', 'orders']
        monthly['month_key'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
        return monthly.to_dict('records')

    def get_category_analysis(self):
        """获取分类分析"""
        cat = self.df.groupby('pizza_category').agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'pizza_name': 'nunique'
        }).reset_index()
        cat.columns = ['category', 'quantity', 'revenue', 'unique_pizzas']
        cat['avg_price'] = round(cat['revenue'] / cat['quantity'], 2)
        cat['revenue_pct'] = round(cat['revenue'] / cat['revenue'].sum() * 100, 2)
        cat['quantity_pct'] = round(cat['quantity'] / cat['quantity'].sum() * 100, 2)
        return cat.to_dict('records')

    def get_size_distribution(self):
        """获取尺寸分布"""
        size = self.df.groupby('pizza_size').agg({
            'quantity': 'sum',
            'total_price': 'sum'
        }).reset_index()
        size.columns = ['size', 'quantity', 'revenue']
        size['revenue_pct'] = round(size['revenue'] / size['revenue'].sum() * 100, 2)
        size['quantity_pct'] = round(size['quantity'] / size['quantity'].sum() * 100, 2)
        return size.to_dict('records')

    def get_top_products(self, limit=10):
        """获取热销产品"""
        top = self.df.groupby('pizza_name').agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'pizza_size': 'count'
        }).reset_index()
        top.columns = ['name', 'quantity', 'revenue', 'order_count']
        top['avg_price'] = round(top['revenue'] / top['quantity'], 2)
        top = top.sort_values('quantity', ascending=False).head(limit)
        return top.to_dict('records')

    def get_heatmap_data(self):
        """获取热力图数据（星期x小时）"""
        heatmap = self.df.groupby(['day_of_week', 'hour']).agg({
            'quantity': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        heatmap.columns = ['day', 'hour', 'quantity', 'orders']

        # 构建热力图矩阵
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hours = list(range(10, 23))

        matrix = []
        for day in range(7):
            row = {'day': days[day], 'data': []}
            for hour in hours:
                cell = heatmap[(heatmap['day'] == day) & (heatmap['hour'] == hour)]
                if len(cell) > 0:
                    row['data'].append({
                        'hour': hour,
                        'quantity': int(cell['quantity'].values[0]),
                        'orders': int(cell['orders'].values[0])
                    })
                else:
                    row['data'].append({
                        'hour': hour,
                        'quantity': 0,
                        'orders': 0
                    })
            matrix.append(row)

        return matrix


class ForecastingService:
    """预测服务"""

    def __init__(self, data_model):
        self.model = data_model
        self.df = data_model.df

    def predict_daily_sales(self, target_date):
        """预测指定日期销量"""
        target_dow = target_date.weekday()
        is_weekend = target_dow in [5, 6]

        # 计算历史同期平均
        historical = self.df[self.df['day_of_week'] == target_dow]
        avg_quantity = historical['quantity'].mean()
        avg_revenue = historical['total_price'].mean()

        # 季节性调整 (简化版)
        month = target_date.month
        seasonal_factors = {
            12: 1.15, 1: 1.15, 2: 1.10,  # 冬季
            3: 1.0, 4: 1.0, 5: 1.05,     # 春季
            6: 1.05, 7: 1.10, 8: 1.10,   # 夏季
            9: 1.0, 10: 1.0, 11: 1.10    # 秋季
        }

        seasonal_factor = seasonal_factors.get(month, 1.0)

        # 周末加成
        weekend_factor = 1.35 if is_weekend else 1.0
        friday_factor = 1.25 if target_dow == 4 else 1.0

        # 综合预测
        predicted_quantity = int(avg_quantity * seasonal_factor * weekend_factor * friday_factor)
        predicted_revenue = int(avg_revenue * seasonal_factor * weekend_factor * friday_factor)

        # 计算置信区间
        std_quantity = historical['quantity'].std()
        confidence_range = std_quantity * 1.96

        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'day_name': target_date.strftime('%A'),
            'is_weekend': is_weekend,
            'predicted_quantity': predicted_quantity,
            'predicted_revenue': predicted_revenue,
            'confidence_low': int(predicted_quantity - confidence_range),
            'confidence_high': int(predicted_quantity + confidence_range),
            'historical_avg': round(avg_quantity, 1)
        }

    def predict_weekly_sales(self, start_date):
        """预测一周销量"""
        predictions = []
        for i in range(7):
            target_date = start_date + timedelta(days=i)
            pred = self.predict_daily_sales(target_date)
            predictions.append(pred)
        return predictions

    def predict_hourly_sales(self, target_date):
        """预测指定日期的小时销量"""
        target_dow = target_date.weekday()
        is_weekend = target_dow in [5, 6]

        # 获取历史小时分布
        hourly = self.df[self.df['day_of_week'] == target_dow].groupby('hour')['quantity'].mean()

        # 预测当天总销量
        daily_pred = self.predict_daily_sales(target_date)

        # 预测各小时销量
        if len(hourly) > 0:
            hourly_total = hourly.sum()
            predictions = []
            for hour in range(10, 23):
                if hour in hourly.index:
                    hour_ratio = hourly[hour] / hourly_total
                    hour_pred = int(daily_pred['predicted_quantity'] * hour_ratio)
                else:
                    hour_pred = 0
                predictions.append({
                    'hour': hour,
                    'predicted_quantity': max(0, hour_pred)
                })
        else:
            # 使用默认分布
            default_pattern = {
                10: 0.06, 11: 0.10, 12: 0.12, 13: 0.09,
                14: 0.05, 15: 0.06, 16: 0.07, 17: 0.10,
                18: 0.14, 19: 0.15, 20: 0.12, 21: 0.08, 22: 0.05
            }
            predictions = []
            for hour in range(10, 23):
                ratio = default_pattern.get(hour, 0)
                hour_pred = int(daily_pred['predicted_quantity'] * ratio)
                predictions.append({
                    'hour': hour,
                    'predicted_quantity': max(0, hour_pred)
                })

        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'total_predicted': daily_pred['predicted_quantity'],
            'hourly_breakdown': predictions
        }


class StaffingService:
    """排班服务"""

    def __init__(self, data_model):
        self.model = data_model
        self.df = data_model.df

    def generate_schedule_recommendations(self, target_date):
        """生成排班建议"""
        target_dow = target_date.weekday()
        is_weekend = target_dow in [5, 6]

        # 获取历史小时订单量
        hourly_orders = self.df[
            self.df['day_of_week'] == target_dow
        ].groupby('hour')['order_id'].nunique().to_dict()

        recommendations = []

        for hour in range(10, 23):
            avg_orders = hourly_orders.get(hour, 5)
            required_staff = max(2, int(np.ceil(avg_orders / StaffingModel.CAPACITY_PER_PERSON)))

            # 周末加成
            if is_weekend:
                required_staff = int(required_staff * StaffingModel.WEEKEND_FACTOR)

            # 高峰额外人手
            if hour in StaffingModel.PEAK_HOURS:
                required_staff += StaffingModel.PEAK_BONUS

            # 压力等级评估
            if avg_orders < 10:
                pressure = 'low'
                pressure_cn = '空闲'
            elif avg_orders < 20:
                pressure = 'medium'
                pressure_cn = '正常'
            elif avg_orders < 30:
                pressure = 'high'
                pressure_cn = '繁忙'
            else:
                pressure = 'critical'
                pressure_cn = '爆满'

            recommendations.append({
                'hour': hour,
                'time_slot': f"{hour:02d}:00-{(hour+1)%24 if hour == 23 else hour+1:02d}:00",
                'avg_orders': round(avg_orders, 1),
                'required_staff': required_staff,
                'pressure': pressure,
                'pressure_cn': pressure_cn,
                'is_peak': hour in StaffingModel.PEAK_HOURS
            })

        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'day_name': target_date.strftime('%A'),
            'is_weekend': is_weekend,
            'recommendations': recommendations,
            'summary': self._generate_schedule_summary(recommendations)
        }

    def _generate_schedule_summary(self, recommendations):
        """生成排班摘要"""
        peak_hours = [r for r in recommendations if r['is_peak']]
        critical_hours = [r for r in recommendations if r['pressure'] == 'critical']

        return {
            'total_hours': len(recommendations),
            'peak_hour_count': len(peak_hours),
            'critical_hour_count': len(critical_hours),
            'avg_staff_required': round(
                np.mean([r['required_staff'] for r in recommendations]), 1
            ),
            'max_staff_needed': max([r['required_staff'] for r in recommendations]),
            'min_staff_needed': min([r['required_staff'] for r in recommendations])
        }

    def generate_weekly_schedule(self, start_date):
        """生成周度排班计划"""
        schedules = []
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

        for i in range(7):
            target_date = start_date + timedelta(days=i)
            schedule = self.generate_schedule_recommendations(target_date)
            schedule['day_index'] = i
            schedule['day_name_cn'] = day_names[i]
            schedules.append(schedule)

        return schedules


class PrepService:
    """备餐服务"""

    def __init__(self, data_model):
        self.model = data_model
        self.df = data_model.df

    def generate_prep_tasks(self, target_date):
        """生成备餐任务"""
        target_dow = target_date.weekday()
        is_weekend = target_dow in [5, 6]

        # 获取历史尺寸分布
        size_dist = self.df.groupby('pizza_size')['quantity'].sum()
        size_ratios = (size_dist / size_dist.sum()).to_dict()

        # 预测当天总销量
        forecasting = ForecastingService(self.model)
        daily_pred = forecasting.predict_daily_sales(target_date)
        total_qty = daily_pred['predicted_quantity']

        tasks = []

        # 生成面团备料任务
        for size, ratio in size_ratios.items():
            predicted_qty = int(total_qty * ratio)
            if predicted_qty > 0:
                dough_kg = predicted_qty * IngredientModel.DOUGH_WEIGHTS.get(size, 0.3)
                tasks.append({
                    'time': '营业前',
                    'task_type': '面团准备',
                    'item': f'{size}面团',
                    'quantity': predicted_qty,
                    'unit': '个',
                    'dough_kg': round(dough_kg, 1),
                    'priority': 'high' if predicted_qty > 30 else 'normal'
                })

        # 生成配料备料任务
        ingredients = ['Mozzarella Cheese', 'Pepperoni', 'Mushrooms', 'Tomatoes', 'Onions']
        for ing in ingredients:
            ing_qty = total_qty * IngredientModel.CONSUMPTION_RATES.get(ing, 0.1)
            tasks.append({
                'time': '10:00',
                'task_type': '配料预处理',
                'item': ing,
                'quantity': round(ing_qty, 1),
                'unit': 'kg',
                'priority': 'high'
            })
            # 下午补货
            tasks.append({
                'time': '16:00',
                'task_type': '配料补充',
                'item': ing,
                'quantity': round(ing_qty * 0.6, 1),
                'unit': 'kg',
                'priority': 'normal'
            })

        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'predicted_sales': total_qty,
            'tasks': tasks,
            'task_count': len(tasks),
            'high_priority_count': len([t for t in tasks if t['priority'] == 'high'])
        }

    def calculate_ingredient_needs(self, target_date):
        """计算原料需求"""
        forecasting = ForecastingService(self.model)
        daily_pred = forecasting.predict_daily_sales(target_date)
        total_qty = daily_pred['predicted_quantity']

        # 按分类分布
        cat_dist = self.df.groupby('pizza_category')['quantity'].sum()
        cat_ratios = (cat_dist / cat_dist.sum()).to_dict()

        # 计算各原料需求
        ingredient_needs = {}
        for ingredient, rate in IngredientModel.CONSUMPTION_RATES.items():
            total_need = total_qty * rate * 1.1  # 10% 损耗系数
            ingredient_needs[ingredient] = round(total_need, 2)

        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'predicted_sales': total_qty,
            'ingredient_needs': ingredient_needs,
            'total_ingredient_kg': round(sum(ingredient_needs.values()), 2)
        }


class InventoryService:
    """库存服务"""

    def __init__(self, data_model):
        self.model = data_model
        self.df = data_model.df
        self.stock_levels = self._init_stock_levels()

    def _init_stock_levels(self):
        """初始化库存水平"""
        return {
            'Flour': {'current': 80, 'unit': 'kg', 'lead_time': 1, 'supplier': '面制品批发商'},
            'Mozzarella Cheese': {'current': 45, 'unit': 'kg', 'lead_time': 1, 'supplier': '乳制品批发'},
            'Tomato Sauce': {'current': 30, 'unit': 'kg', 'lead_time': 2, 'supplier': '调料批发商'},
            'Pepperoni': {'current': 25, 'unit': 'kg', 'lead_time': 1, 'supplier': '肉类供应商'},
            'Mushrooms': {'current': 20, 'unit': 'kg', 'lead_time': 1, 'supplier': '蔬菜批发'},
            'Chicken': {'current': 15, 'unit': 'kg', 'lead_time': 1, 'supplier': '鸡肉供应商'},
            'Bacon': {'current': 12, 'unit': 'kg', 'lead_time': 1, 'supplier': '肉类供应商'},
            'Onions': {'current': 18, 'unit': 'kg', 'lead_time': 1, 'supplier': '蔬菜批发'},
            'Green Peppers': {'current': 15, 'unit': 'kg', 'lead_time': 1, 'supplier': '蔬菜批发'},
            'Olives': {'current': 10, 'unit': 'kg', 'lead_time': 2, 'supplier': '调料批发商'}
        }

    def get_inventory_alerts(self, target_date):
        """获取库存预警"""
        prep_service = PrepService(self.model)
        needs = prep_service.calculate_ingredient_needs(target_date)
        ingredient_needs = needs['ingredient_needs']

        # 计算每日消耗
        daily_consumption = {}
        for ing in self.stock_levels.keys():
            daily_consumption[ing] = ingredient_needs.get(ing, 5)

        alerts = []

        for ingredient, data in self.stock_levels.items():
            daily_rate = daily_consumption.get(ingredient, 5)
            days_remaining = data['current'] / daily_rate if daily_rate > 0 else 999

            # 计算建议补货量
            recommended_order = int(daily_rate * 5)

            # 预警等级
            if days_remaining < 1:
                level = 'critical'
                level_cn = '紧急'
                message = f'库存告急！{ingredient}将在{days_remaining:.1f}天内用完，立即补货！'
            elif days_remaining < 2:
                level = 'warning'
                level_cn = '警告'
                message = f'{ingredient}库存偏低，剩余{days_remaining:.1f}天用量，今晚需补货。'
            elif days_remaining < 3:
                level = 'notice'
                level_cn = '提醒'
                message = f'{ingredient}库存尚可，建议明日补货 {recommended_order}{data["unit"]}。'
            else:
                level = 'ok'
                level_cn = '正常'
                message = f'{ingredient}库存充足，预计可使用{days_remaining:.1f}天。'

            alerts.append({
                'ingredient': ingredient,
                'current_stock': data['current'],
                'unit': data['unit'],
                'daily_consumption': round(daily_rate, 2),
                'days_remaining': round(days_remaining, 1),
                'recommended_order': recommended_order,
                'supplier': data['supplier'],
                'lead_time': data['lead_time'],
                'level': level,
                'level_cn': level_cn,
                'message': message
            })

        # 按紧急程度排序
        level_order = {'critical': 0, 'warning': 1, 'notice': 2, 'ok': 3}
        alerts.sort(key=lambda x: level_order[x['level']])

        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'alerts': alerts,
            'summary': {
                'total_items': len(alerts),
                'critical_count': len([a for a in alerts if a['level'] == 'critical']),
                'warning_count': len([a for a in alerts if a['level'] == 'warning']),
                'notice_count': len([a for a in alerts if a['level'] == 'notice']),
                'ok_count': len([a for a in alerts if a['level'] == 'ok'])
            }
        }

    def update_stock(self, ingredient, quantity):
        """更新库存"""
        if ingredient in self.stock_levels:
            self.stock_levels[ingredient]['current'] = quantity
            return {'success': True, 'message': f'{ingredient}库存已更新为{quantity}'}
        return {'success': False, 'message': f'未找到{ingredient}'}


class WeatherService:
    """天气分析服务"""

    def __init__(self, data_model):
        self.model = data_model
        self.df = data_model.df

    def analyze_weather_impact(self, weather_type, temperature):
        """分析天气影响"""
        # 获取基础销量
        avg_daily = self.df['quantity'].sum() / 365

        # 确定天气影响系数
        if temperature > 35:
            impact_factor = WeatherImpactModel.IMPACT_COEFFICIENTS['hot']
            weather_desc = '高温天气'
        elif temperature < 5:
            impact_factor = WeatherImpactModel.IMPACT_COEFFICIENTS['cold']
            weather_desc = '寒冷天气'
        elif weather_type in ['rain', 'drizzle']:
            impact_factor = WeatherImpactModel.IMPACT_COEFFICIENTS['rainy']
            weather_desc = '雨天'
        elif weather_type == 'heavy_rain':
            impact_factor = WeatherImpactModel.IMPACT_COEFFICIENTS['heavy_rain']
            weather_desc = '大雨'
        elif weather_type in ['snow', 'sleet']:
            impact_factor = WeatherImpactModel.IMPACT_COEFFICIENTS['snowy']
            weather_desc = '雪天'
        elif weather_type == 'cloudy':
            impact_factor = WeatherImpactModel.IMPACT_COEFFICIENTS['cloudy']
            weather_desc = '阴天'
        else:
            impact_factor = WeatherImpactModel.IMPACT_COEFFICIENTS['sunny']
            weather_desc = '晴天'

        predicted_sales = int(avg_daily * impact_factor)

        return {
            'weather_type': weather_type,
            'weather_desc': weather_desc,
            'temperature': temperature,
            'impact_factor': impact_factor,
            'predicted_daily_sales': predicted_sales,
            'sales_change_pct': round((impact_factor - 1) * 100, 1),
            'recommendations': self._generate_weather_recommendations(weather_type, temperature)
        }

    def _generate_weather_recommendations(self, weather_type, temperature):
        """生成天气应对建议"""
        recommendations = []

        if temperature > 35:
            recommendations.append({
                'type': 'prep',
                'message': '减少热饮备货量30%，增加冷饮原料'
            })
            recommendations.append({
                'type': 'promo',
                'message': '推出"冰爽夏日套餐"'
            })

        if weather_type in ['rain', 'heavy_rain', 'snow']:
            recommendations.append({
                'type': 'staff',
                'message': '外卖需求激增，建议增加配送人员'
            })
            recommendations.append({
                'type': 'promo',
                'message': '推出"雨天特惠套餐"'
            })

        if temperature < 5:
            recommendations.append({
                'type': 'prep',
                'message': '增加热饮和热食备货量'
            })

        return recommendations


class PromotionService:
    """促销服务"""

    def __init__(self, data_model):
        self.model = data_model
        self.df = data_model.df

    def calculate_price_elasticity(self, category=None):
        """计算价格弹性"""
        if category:
            df = self.df[self.df['pizza_category'] == category]
            base_elasticity = WeatherImpactModel.PRICE_ELASTICITY.get(category, -1.5)
        else:
            df = self.df
            base_elasticity = -1.5

        # 根据周末/工作日调整
        weekend_factor = 0.8 if df['is_weekend'].mean() > 0.3 else 1.0

        return {
            'category': category or 'all',
            'base_elasticity': base_elasticity,
            'weekend_elasticity': round(base_elasticity * weekend_factor, 2),
            'interpretation': self._interpret_elasticity(base_elasticity)
        }

    def _interpret_elasticity(self, elasticity):
        """解释价格弹性"""
        if elasticity < -1.5:
            return '高度敏感，适合降价促销'
        elif elasticity < -1:
            return '中度敏感，可适度调整价格'
        elif elasticity < -0.5:
            return '低度敏感，可考虑提价'
        else:
            return '不敏感，价格调整影响较小'

    def simulate_promotion(self, promotion_type, base_sales, category=None):
        """模拟促销效果"""
        effect = PromotionModel.PROMOTION_EFFECTS.get(promotion_type, {
            'sales_lift': 0.20,
            'margin_impact': -0.10
        })

        # 计算当前毛利 (简化估算)
        current_revenue = base_sales * self.df['total_price'].mean()
        current_margin = current_revenue * 0.35  # 假设35%毛利

        # 促销后
        new_sales = int(base_sales * (1 + effect['sales_lift']))
        new_revenue = new_sales * self.df['total_price'].mean() * (1 + effect['margin_impact'])
        new_margin = new_revenue * 0.35

        return {
            'promotion_type': promotion_type,
            'base_sales': base_sales,
            'predicted_sales': new_sales,
            'sales_increase': new_sales - base_sales,
            'sales_lift_pct': round(effect['sales_lift'] * 100, 1),
            'margin_impact_pct': round(effect['margin_impact'] * 100, 1),
            'profit_change': round(new_margin - current_margin, 2),
            'roi_estimate': 'positive' if new_margin > current_margin else 'negative'
        }

    def get_promotion_recommendations(self, conditions):
        """获取促销建议"""
        recommendations = []

        # 基于天气
        if conditions.get('weather') == 'rainy':
            recommendations.append({
                'type': 'combo',
                'title': '雨天外卖套餐',
                'description': '推出披萨+饮品组合，刺激外卖需求',
                'expected_lift': '+25%'
            })

        # 基于周末
        if conditions.get('is_weekend'):
            recommendations.append({
                'type': 'family',
                'title': '家庭聚餐优惠',
                'description': '大份披萨第二份半价',
                'expected_lift': '+30%'
            })

        # 基于月末
        if conditions.get('is_month_end'):
            recommendations.append({
                'type': 'clearance',
                'title': '月末清仓促销',
                'description': '指定产品8折优惠',
                'expected_lift': '+35%'
            })

        return recommendations


# ============================================================================
# API 路由
# ============================================================================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/summary')
def api_summary():
    """数据摘要API"""
    summary = get_data_summary()
    if summary:
        return jsonify({'success': True, 'data': summary})
    return jsonify({'success': False, 'message': '数据加载失败'})


@app.route('/api/analysis/hourly')
def api_hourly_pattern():
    """小时模式分析API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    service = SalesAnalysisService(model)
    data = service.get_hourly_pattern()
    return jsonify({'success': True, 'data': data})


@app.route('/api/analysis/weekday')
def api_weekday_pattern():
    """星期模式分析API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    service = SalesAnalysisService(model)
    data = service.get_weekday_pattern()
    return jsonify({'success': True, 'data': data})


@app.route('/api/analysis/category')
def api_category_analysis():
    """分类分析API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    service = SalesAnalysisService(model)
    data = service.get_category_analysis()
    return jsonify({'success': True, 'data': data})


@app.route('/api/analysis/size')
def api_size_distribution():
    """尺寸分布API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    service = SalesAnalysisService(model)
    data = service.get_size_distribution()
    return jsonify({'success': True, 'data': data})


@app.route('/api/analysis/top-products')
def api_top_products():
    """热销产品API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    service = SalesAnalysisService(model)
    limit = request.args.get('limit', 10, type=int)
    data = service.get_top_products(limit)
    return jsonify({'success': True, 'data': data})


@app.route('/api/analysis/heatmap')
def api_heatmap():
    """热力图数据API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    service = SalesAnalysisService(model)
    data = service.get_heatmap_data()
    return jsonify({'success': True, 'data': data})


@app.route('/api/forecast/daily')
def api_forecast_daily():
    """日预测API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    date_str = request.args.get('date')
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    service = ForecastingService(model)
    data = service.predict_daily_sales(target_date)
    return jsonify({'success': True, 'data': data})


@app.route('/api/forecast/weekly')
def api_forecast_weekly():
    """周预测API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    date_str = request.args.get('start_date')
    if date_str:
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        start_date = datetime.now()

    service = ForecastingService(model)
    data = service.predict_weekly_sales(start_date)
    return jsonify({'success': True, 'data': data})


@app.route('/api/forecast/hourly')
def api_forecast_hourly():
    """小时预测API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    date_str = request.args.get('date')
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    service = ForecastingService(model)
    data = service.predict_hourly_sales(target_date)
    return jsonify({'success': True, 'data': data})


@app.route('/api/schedule/daily')
def api_schedule_daily():
    """日排班建议API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    date_str = request.args.get('date')
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    service = StaffingService(model)
    data = service.generate_schedule_recommendations(target_date)
    return jsonify({'success': True, 'data': data})


@app.route('/api/schedule/weekly')
def api_schedule_weekly():
    """周排班计划API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    date_str = request.args.get('start_date')
    if date_str:
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        start_date = datetime.now()

    service = StaffingService(model)
    data = service.generate_weekly_schedule(start_date)
    return jsonify({'success': True, 'data': data})


@app.route('/api/prep/tasks')
def api_prep_tasks():
    """备餐任务API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    date_str = request.args.get('date')
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    service = PrepService(model)
    data = service.generate_prep_tasks(target_date)
    return jsonify({'success': True, 'data': data})


@app.route('/api/prep/ingredients')
def api_ingredient_needs():
    """原料需求API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    date_str = request.args.get('date')
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    service = PrepService(model)
    data = service.calculate_ingredient_needs(target_date)
    return jsonify({'success': True, 'data': data})


@app.route('/api/inventory/alerts')
def api_inventory_alerts():
    """库存预警API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    date_str = request.args.get('date')
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    service = InventoryService(model)
    data = service.get_inventory_alerts(target_date)
    return jsonify({'success': True, 'data': data})


@app.route('/api/weather/impact')
def api_weather_impact():
    """天气影响分析API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    weather_type = request.args.get('type', 'sunny')
    temperature = request.args.get('temperature', 20, type=float)

    service = WeatherService(model)
    data = service.analyze_weather_impact(weather_type, temperature)
    return jsonify({'success': True, 'data': data})


@app.route('/api/promotion/elasticity')
def api_price_elasticity():
    """价格弹性API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    category = request.args.get('category')

    service = PromotionService(model)
    data = service.calculate_price_elasticity(category)
    return jsonify({'success': True, 'data': data})


@app.route('/api/promotion/simulate')
def api_promotion_simulate():
    """促销效果模拟API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    promotion_type = request.args.get('type', 'discount_20')
    base_sales = request.args.get('base_sales', 100, type=int)
    category = request.args.get('category')

    service = PromotionService(model)
    data = service.simulate_promotion(promotion_type, base_sales, category)
    return jsonify({'success': True, 'data': data})


@app.route('/api/promotion/recommendations')
def api_promotion_recommendations():
    """促销建议API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    conditions = {
        'weather': request.args.get('weather'),
        'is_weekend': request.args.get('is_weekend', 'false').lower() == 'true',
        'is_month_end': request.args.get('is_month_end', 'false').lower() == 'true'
    }

    service = PromotionService(model)
    data = service.get_promotion_recommendations(conditions)
    return jsonify({'success': True, 'data': data})


@app.route('/api/export/all')
def api_export_all():
    """导出所有数据API"""
    model = load_sales_data()
    if model is None:
        return jsonify({'success': False, 'message': '数据加载失败'})

    # 收集所有数据
    export_data = {
        'summary': get_data_summary(),
        'hourly_pattern': SalesAnalysisService(model).get_hourly_pattern(),
        'weekday_pattern': SalesAnalysisService(model).get_weekday_pattern(),
        'category_analysis': SalesAnalysisService(model).get_category_analysis(),
        'size_distribution': SalesAnalysisService(model).get_size_distribution(),
        'top_products': SalesAnalysisService(model).get_top_products(20),
        'weekly_forecast': ForecastingService(model).predict_weekly_sales(datetime.now()),
        'weekly_schedule': StaffingService(model).generate_weekly_schedule(datetime.now()),
        'inventory_alerts': InventoryService(model).get_inventory_alerts(datetime.now()),
        'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 保存到文件
    export_path = os.path.join(DATA_DIR, 'export_data.json')
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

    return jsonify({
        'success': True,
        'message': '数据已导出',
        'path': export_path
    })


# ============================================================================
# 错误处理
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'API不存在'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'success': False, 'message': '服务器内部错误'}), 500


# ============================================================================
# 启动应用
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("披萨店智能运营系统 (Pizza Smart Ops)")
    print("Pizza Store Intelligent Operations System")
    print("=" * 60)
    print("\n启动服务...")
    print("访问地址: http://localhost:5002")
    print("\n按 Ctrl+C 停止服务\n")

    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True
    )
