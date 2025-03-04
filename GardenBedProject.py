import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Project phases with schedule, cost, risks, and time dependency
phases = [
    ('Soil Testing', 7, 20, 'Inaccurate results', 'TI'),
    ('Site Preparation', 4, 450, 'Weather delays', 'TD'),
    ('Material Gathering', 4, 300, 'Material availability', 'TI'),
    ('Constructing Garden Bed', 5, 15, 'Tool issues', 'TD'),
    ('Irrigation Setup', 2, 70, 'Installation issues', 'TD'),
    ('Seeding Setup', 2, 450, 'Plant supply issues', 'TI')
]

# Start date of the project for beginning of Feb 2025
start_date = datetime(2025,2,1)

# Calculate start and end dates for each phase with dependencies and overlaps
schedule = []
np.random.seed(42)

# Define start times and dependencies
soil_testing_start = start_date
site_prep_start = soil_testing_start + timedelta(days=2) # Overlaps soil testing after 2 days
material_gathering_start = soil_testing_start + timedelta(days=7) # Waits until soil testing is done
construction_start = material_gathering_start + timedelta(days=4)
irrigation_start = construction_start + timedelta(days=5)
seeding_start = irrigation_start + timedelta(days=2)

starts = [
    soil_testing_start,
    site_prep_start,
    material_gathering_start,
    construction_start,
    irrigation_start,
    seeding_start
]

for (phase, duration, cost, risk, cost_type), start in zip(phases, starts):
    end_date = start + timedelta(days=duration)
    schedule.append((phase, start, end_date, duration, cost, risk, cost_type))

# Create a Gantt chart for the updated schedule
plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0.3, right=0.85)
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']


for i, (phase, start, end, duration, cost, risk, cost_type) in enumerate(schedule):
    plt.barh(phase, duration, left=(start - start_date).days, color=colors[i % len(colors)], edgecolor='black', height=0.8)
    plt.text((start - start_date).days + duration / 2, i, f'{start.date()} to {end.date()}', ha='center', va='bottom', color='black', fontsize=10, weight='bold')

plt.xlabel('Days from Start')
plt.title('Garden Bed Project Schedule')
plt.grid(True)

# Define the vegetable planting and harvesting schedules with integrated replanting for succession crops
vegetable_schedule = [
    ('Tomatoes', '2025-04-15', '2025-09-01', []),
    ('Peppers', '2025-04-15', '2025-10-01', []),
    ('Cucumbers', '2025-05-01', '2025-09-01', []),
    ('Green Beans', '2025-04-01', '2025-10-01', ['2025-04-22', '2025-05-13']),
    ('Squash', '2025-05-01', '2025-09-01', []),
    ('Carrots', '2025-03-15', '2025-07-01', ['2025-04-05', '2025-04-26']),
    ('Lettuce', '2025-03-15', '2025-06-01', ['2025-03-29', '2025-04-12']),
    ('Onions', '2025-03-01', '2025-07-01', [])
]

# Convert dates to datetime objects
start_date = datetime(2025,2,1)
vegetable_schedule = [(veg, datetime.strptime(start, '%Y-%m-%d'), datetime.strptime(end, '%Y-%m-%d'), [datetime.strptime(replant, '%Y-%m-%d') for replant in replants])
                      for veg, start, end, replants in vegetable_schedule]

# Create a new Gantt chart for planting and harvesting with different colors
plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0.3, right=0.85)
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink', 'cyan', 'magenta']

for i, (veg, start, end, replants) in enumerate(vegetable_schedule):
    duration = (end - start).days
    plt.barh(veg, duration, left=(start - start_date).days, color=colors[i % len(colors)], edgecolor='black', height=0.8)
    plt.text((start - start_date).days + duration / 2, i, f'{start.date()} to {end.date()}', ha='center', va='bottom', color='black', fontsize=8, weight='bold')
    
    # Add arrows or markers for replanting dates
    for replant in replants:
        plt.text((replant - start_date).days, i, 'R', ha='center', va='top', color='black', fontsize=8, weight='bold')

plt.xlabel('Days from Start')
plt.title('Vegetable Planting and Harvesting Timeline with Replanting Markers')
plt.grid(False)

# Create folder for outputs
output_folder = r'C:\Users\DawsonMcClary\OneDrive - Liberty Oilfield Services\Desktop\School'
os.makedirs(output_folder, exist_ok=True)

#Save plots of garden schedule
plt.savefig(os.path.join(output_folder, 'garden_project_plants_plot.png'))

# Simulated risk impact and probability values
risk_impact = np.random.randint(1, 3, len(phases))
risk_probability = np.random.randint(10, 50, len(phases))

# Plot enhanced Gantt chart with dependencies, risk percentages, and add legend (edited outside)
plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0.3, right=0.85)

blue_patch = plt.Line2D([0], [0], color = 'blue', lw=4, label='TI')
green_patch = plt.Line2D([0], [0], color = 'green', lw=4, label='TD')

for i, (phase, start, end, duration, cost, risk, cost_type), probability in zip(range(len(schedule)), schedule, risk_probability):
    color = 'blue' if cost_type == 'TI' else 'green'
    plt.barh(phase, duration, left=(start - start_date).days, color=color, edgecolor='black', height=0.8)
    plt.text((start - start_date).days + duration / 2, i, f'', ha='center', va='bottom', color='black', fontsize=9, weight='bold')

plt.legend(handles=[blue_patch, green_patch], loc='upper left')
plt.xlabel('Days from Start')
plt.title('Backyard Garden Bed Project Schedule with Dependencies, Risks, and Uncertainty')
plt.grid(True)

# Save plot to School folder
plt.savefig(os.path.join(output_folder, 'garden_project_schedule_plot.png'))

# Print project schedule with costs, risks, and cost type in a table format and write to a text file
output = []
output.append("\nProject Schedule, Costs, and Risks:")
output.append(f"{'Phase':<25}{'Start Date':<12}{'End Date':<12}{'Duration (days)':<15}{'Cost ($)':<10}{'Cost Type':<10}{'Potential Risks':<20}")
output.append("-" * 110)
for phase, start, end, duration, cost, risk, cost_type in schedule:
    output.append(f"{phase:<25}{start.date():<12}{end.date():<12}{duration:<15}{cost:<10}${cost_type:<10}{risk:<20}")

# Write schedule to text file
with open(os.path.join(output_folder, 'garden_project_schedule.txt'), 'w') as f:
    for line in output:
        f.write(line + '\n')

# Risk analysis breakdown
risk_analysis = []
risk_analysis.append("\nRisk Analysis for Garden Project:")
risk_analysis.append(f"{'Phase':<25}{'Potential Risks':<30}{'Risk Impact (1-5)':<20}{'Risk Probability (%)':<20}")
risk_analysis.append("-" * 100)

for (phase, _, _, risk, _), impact, probability in zip(phases, risk_impact, risk_probability):
    risk_analysis.append(f"{phase:<25}{risk:<30}{impact:<20}{probability:<20}")

# Write risk analysis to text file
with open(os.path.join(output_folder, 'garden_project_risk_analysis.txt'), 'w') as f:
    for line in risk_analysis:
        f.write(line + '\n')

# Monte Carlo Simulation for Project Cost and Schedule Confidence with Variability
confidence_levels = [0.05, 0.5, 0.75, 0.95, 0.99]

# Risk factors - variability (as percentage)
cost_variability = 0.1  # 10% variability
schedule_variability = 0.15  # 15% variability

cost_simulation = np.random.normal(loc=1300, scale=1300 * cost_variability, size=1000)
schedule_simulation = np.random.normal(loc=20, scale=20 * schedule_variability, size=1000)

jcl_results = {}
cost_thresholds = {}
schedule_thresholds = {}

colors = ['purple', 'orange', 'cyan', 'magenta', 'yellow']

for cl in confidence_levels:
    cost_threshold = np.percentile(cost_simulation, cl * 100)
    schedule_threshold = np.percentile(schedule_simulation, cl * 100)
    jcl_results[cl] = np.mean((cost_simulation <= cost_threshold) & (schedule_simulation <= schedule_threshold))
    cost_thresholds[cl] = cost_threshold
    schedule_thresholds[cl] = schedule_threshold

# Print and write JCL results to text file
jcl_output = []
jcl_output.append("\nJCL Analysis Results:")
for cl, jcl in jcl_results.items():
    jcl_output.append(f"Confidence Level: {cl*100}% - JCL: {jcl*100:.2f}%")
    jcl_output.append(f"  Cost Threshold: ${cost_thresholds[cl]:,.2f}")
    jcl_output.append(f"  Schedule Threshold: {schedule_thresholds[cl]:.2f} days")

# Write JCL analysis to text file
with open(os.path.join(output_folder, 'garden_project_jcl_analysis.txt'), 'w') as f:
    for line in jcl_output:
        f.write(line + '\n')

# Annotate 75% confidence level on plots
seventy_five_cl = 0.75

# Plot JCL results
plt.figure(figsize=(12, 8))

# Plot cost distribution
plt.subplot(2, 1, 1)
plt.hist(cost_simulation, bins=50, alpha=0.75, color='blue')
plt.axvline(np.mean(cost_simulation), color='red', linestyle='dashed', linewidth=2, label='Mean Cost')
for cl, col in zip(confidence_levels, colors):
    plt.axvline(cost_thresholds[cl], linestyle='--', color=col, label=f'{cl*100}% Threshold')
plt.title('Project Cost Simulation for Garden Bed 2025')
plt.xlabel('Cost ($)')
plt.ylabel('Frequency')
plt.legend()

# Annotate 75% cost threshold
plt.annotate(f'75% Threshold: ${cost_thresholds[seventy_five_cl]:,.2f}',
             xy=(cost_thresholds[seventy_five_cl], 20),
             xytext=(cost_thresholds[seventy_five_cl] + 50, 25),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10, color='black')

# Plot schedule distribution
plt.subplot(2, 1, 2)
plt.hist(schedule_simulation, bins=50, alpha=0.75, color='green')
plt.axvline(np.mean(schedule_simulation), color='red', linestyle='dashed', linewidth=2, label='Mean Schedule')
for cl, col in zip(confidence_levels, colors):
    plt.axvline(schedule_thresholds[cl], linestyle='--', color=col, label=f'{cl*100}% Threshold')
plt.title('Project Schedule Simulation for Garden Bed 2025')
plt.xlabel('Schedule (Days)')
plt.ylabel('Frequency')
plt.legend()

# Annotate 75% schedule threshold
plt.annotate(f'75% Threshold: {schedule_thresholds[seventy_five_cl]:.2f} days',
             xy=(schedule_thresholds[seventy_five_cl], 20),
             xytext=(schedule_thresholds[seventy_five_cl] + 1, 25),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10, color='black')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'garden_project_jcl_plot_annotated.png'))

# Garden parameters
garden_size_sqft = 20 * 20  # 20ft x 20ft
garden_size_acres = garden_size_sqft / 43560  # Convert to acres

# Average water needs per week (in inches)
average_weekly_water_need = 1.5

# Estimated number of growing weeks (Feb to Nov)
growing_weeks = 40

# Monthly average rainfall in Oklahoma (in inches)
rainfall_per_month = {
    'Feb': 1.8, 'Mar': 3.1, 'Apr': 3.5, 'May': 5.3, 'Jun': 3.9,
    'Jul': 2.6, 'Aug': 2.9, 'Sep': 4.1, 'Oct': 3.6, 'Nov': 2.8
}

# Total rainfall over the growing season (Feb to Nov)
total_rainfall = sum(rainfall_per_month.values())

# Total water needed without rainfall (in inches)
total_water_inches = average_weekly_water_need * growing_weeks

# Adjust for rainfall
total_water_inches -= total_rainfall

# Convert inches of water to gallons (1 inch over 1 acre = 27,154 gallons)
total_gallons = total_water_inches * 27154 * garden_size_acres

print(f"Total estimated water usage: {total_gallons:.2f} gallons")

#calculate water costs in Oklahoma

def calculate_annual_water_cost(annual_gallons, rate_per_1000_gallons, monthly_base_fee, months_active):
    """
    Calculate the annual water cost for a garden based on water usage and rate.

    Parameters:
    annual_gallons (float): Total gallons of water used annually.
    rate_per_1000_gallons (float): Cost per 1,000 gallons of water.
    monthly_base_fee (float): Fixed monthly service fee.
    months_active (int): Number of months the water is used.

    Returns:
    float: Estimated total annual water cost.
    """
    # Calculate monthly water usage
    monthly_gallons = annual_gallons / months_active

    # Convert monthly gallons to units of 1,000 gallons, rounding up
    monthly_units = -(-monthly_gallons // 1000)  # Equivalent to math.ceil(monthly_gallons / 1000)

    # Calculate monthly variable water cost based on usage
    monthly_variable_cost = monthly_units * rate_per_1000_gallons

    # Calculate total monthly cost
    total_monthly_cost = monthly_variable_cost + monthly_base_fee

    # Calculate total annual cost for the active months
    total_annual_cost = total_monthly_cost * months_active

    return round(total_annual_cost, 2)

if __name__ == "__main__":
    # Inputs
    annual_gallons = total_gallons  # Total gallons of water used per year
    rate_per_1000_gallons = 8.00  # Estimated cost per 1,000 gallons in USD
    monthly_base_fee = 0  # Estimated monthly base service fee in USD (not included)
    months_active = 10  # Number of months water is used (February to November)

    # Calculate and display annual water cost
    total_cost = calculate_annual_water_cost(annual_gallons, rate_per_1000_gallons, monthly_base_fee, months_active)
    print(f"Estimated annual water cost: ${total_cost}")

# Project parameters
average_data = {'initial': 1500, 'monthly': 20, 'return': 1400}
max_data = {'initial': 1400, 'monthly': 15, 'return': 1500}
min_data = {'initial': 1600, 'monthly': 30, 'return': 1300}
months = 10

discount_rate = 0.05 / 12  # 5% annual discount rate, converted to monthly

# Function to calculate cash flows
def calculate_cash_flows(data):
    cash_flows = []
    for year in range(5):
        setup_cost = data['initial'] if year == 0 else data['initial'] / 2
        annual_cash_flow = [-setup_cost] + [-data['monthly']] * (months - 1) + [data['return']]
        cash_flows.extend(annual_cash_flow)
    return cash_flows

# Cash flows for each scenario
average_cash_flows = calculate_cash_flows(average_data)
max_cash_flows = calculate_cash_flows(max_data)
min_cash_flows = calculate_cash_flows(min_data)

# NPV calculation
rates = np.linspace(0.01, 0.15, 500)
average_npvs = [sum(cf / (1 + r / 12) ** t for t, cf in enumerate(average_cash_flows)) for r in rates]
max_npvs = [sum(cf / (1 + r / 12) ** t for t, cf in enumerate(max_cash_flows)) for r in rates]
min_npvs = [sum(cf / (1 + r / 12) ** t for t, cf in enumerate(min_cash_flows)) for r in rates]

# Plotting NPV Sensitivity Analysis
plt.figure(figsize=(8, 6))
sns.lineplot(x=rates * 100, y=average_npvs, color='green', label='Average NPV')
sns.lineplot(x=rates * 100, y=max_npvs, color='blue', label='Max NPV')
sns.lineplot(x=rates * 100, y=min_npvs, color='red', label='Min NPV')

# Break-even rate
breakeven_rate = rates[np.argmin(np.abs(average_npvs))] * 100
plt.axvline(x=breakeven_rate, color='blue', linestyle='--', linewidth=1.5)
plt.annotate('Break-even Point', xy=(breakeven_rate, 0), xytext=(breakeven_rate - 5, 1000),
             arrowprops=dict(facecolor='blue', arrowstyle='->'), fontsize=10, color='purple')

plt.legend()
plt.title('NPV Sensitivity Analysis Across Discount Rates')
plt.xlabel('Discount Rate (%)')
plt.ylabel('Net Present Value ($)')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'garden_project_NVP.png'))
plt.show()

import matplotlib.pyplot as plt

def calculate_cash_flow(initial_cost, annual_expenses, annual_return, years):
    cash_flow = []
    cumulative_cash_flow = -initial_cost  # Start with the initial investment at year 0
    cash_flow.append(cumulative_cash_flow)

    for year in range(1, years + 1):
        if year == 1:
            annual_expense = annual_expenses  # First-year expenses
        else:
            annual_expense = 750  # Fixed annual expense after the first year

        annual_cash_flow = annual_return - annual_expense
        cumulative_cash_flow += annual_cash_flow
        cash_flow.append(cumulative_cash_flow)

    return cash_flow


def calculate_payback_period(cash_flow):
    for year, balance in enumerate(cash_flow, start=0):
        if balance >= 0:
            return year - 1 + (0 - cash_flow[year - 1]) / (balance - cash_flow[year - 1])

    return None


def plot_cash_flow(cash_flow):
    plt.figure(figsize=(8, 6))
    plt.plot(range(0, len(cash_flow)), cash_flow, marker='o', linestyle='-', color='green', label='Cumulative Cash Flow')
    plt.axhline(0, color='red', linestyle='--', label='Break-even Line')
    plt.bar(range(0, len(cash_flow)), cash_flow, color='lightgreen', alpha=0.5, label='Annual Cash Flow')
    plt.title('Garden Project Payback Period over 5 Years')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Cash Flow ($)')
    plt.xticks(range(0, len(cash_flow)))
    plt.legend()
    plt.grid(True)
    plt.show()


initial_cost = 1500
annual_expenses = 750
annual_return = 1400
years = 5

cash_flow = calculate_cash_flow(initial_cost, annual_expenses, annual_return, years)
payback_period = calculate_payback_period(cash_flow)

print(f"Payback period: {payback_period:.2f} years")

plot_cash_flow(cash_flow)

