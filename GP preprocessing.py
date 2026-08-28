import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def LoadDataset(trainPath, testPath, rulPath):

    # Load Excel turbofan engine datasets with headers (ID, Cycle, S1-S21)

    # Read Excel files with headers in first row
    train = pd.read_excel(trainPath, header=0)
    test = pd.read_excel(testPath, header=0)
    rul = pd.read_excel(rulPath, header=0)

    print("Excel datasets loaded WITH HEADERS!")
    print(f"  Train columns: {list(train.columns)[:5]}...")
    print(f"  Train shape: {train.shape}")
    print(f"  Test shape: {test.shape}")
    print(f"  RUL shape: {rul.shape}")

    return train, test, rul

def CheckDataIntegrity(DataFrame, DataFrameName):
    # Comprehensive data quality check for dataset validation
    print(f"\n--------------------------------------------- Checking {DataFrameName} --------------------------------------------\n")
    print("Columns:", list(DataFrame.columns))

    # Check 1: Missing values per column
    missing = DataFrame.isnull().sum()
    print("Missing values per column:")
    if missing[missing > 0].empty:
        print("No missing values found!")
    else:
        print(missing[missing > 0])

    # Check 2: Duplicate rows count
    duplicates = DataFrame.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    # Check 3: Basic statistics summary
    print("\nBasic stats summary:")
    print(DataFrame.describe())

    # Check 4: Outlier detection for sensor columns only (exclude RUL)
    SensorCols = [col for col in DataFrame.columns if col.startswith('S')]
    if 'RUL' in DataFrame.columns:
        SensorCols = [col for col in SensorCols if col != 'RUL']

    print("\nOutlier detection by IQR for sensor columns:")
    for col in SensorCols:
        Q1 = DataFrame[col].quantile(0.25)
        Q3 = DataFrame[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = DataFrame[(DataFrame[col] < Q1 - 1.5*IQR) | (DataFrame[col] > Q3 + 1.5*IQR)][col]
        perc = 100 * len(outliers) / len(DataFrame)
        print(f"{col}: {len(outliers)} outliers ({perc:.2f}%)")

def CleanDataBasic(DataFrame, sensorsToKeep=None):
    # Complete data cleaning pipeline for turbofan sensor data
    print(f"Original columns: {list(DataFrame.columns)}")

    # Step 1: Identify columns to keep
    cols_to_select = ['ID', 'Cycle']

    has_rul = 'RUL' in DataFrame.columns
    if has_rul:
        cols_to_select.append('RUL')
        print("Extracting ID, Cycle, S1-S21 columns and RUL")
    else:
        print("Extracting ID, Cycle, S1-S21 columns")

    # Add all sensor columns initially to allow for low variance removal if sensorsToKeep is None
    initialSensorCols = [col for col in DataFrame.columns if col.startswith('S')]
    cols_to_select.extend(initialSensorCols)

    DataFrameClean = DataFrame[cols_to_select].copy()

    # Step 2: Remove duplicate rows
    initialRows = len(DataFrameClean)
    DataFrameClean = DataFrameClean.drop_duplicates()
    print(f"Dropped {initialRows - len(DataFrameClean)} duplicates")

    # Step 3: Forward/backward fill missing values
    DataFrameClean = DataFrameClean.ffill().bfill()

    # Step 4: Sensor selection (remove low variance or use specified sensors)
    currentSensorCols = [col for col in DataFrameClean.columns if col.startswith('S')]

    if sensorsToKeep is None:
        # Dynamic removal of low variance sensors (variance < 0.001)
        variances = DataFrameClean[currentSensorCols].var()
        lowVarSensors = variances[variances < 0.001].index.tolist()
        if lowVarSensors:
            print(f"Dropping low variance sensors: {lowVarSensors}")
            DataFrameClean = DataFrameClean.drop(columns=lowVarSensors)
            currentSensorCols = [col for col in currentSensorCols if col not in lowVarSensors]
        else:
            print("No low variance sensors found!")
    else:
        # Use specified sensor columns
        validSensorsToKeep = [s for s in sensorsToKeep if s in DataFrameClean.columns]
        print(f"Keeping specified sensors: {validSensorsToKeep}")
        # Re-select to ensure only specified sensors, ID, Cycle, and RUL (if present) are kept
        final_selection_cols = ['ID', 'Cycle'] + validSensorsToKeep
        if has_rul:
            final_selection_cols.append('RUL')
        DataFrameClean = DataFrameClean[final_selection_cols].copy()
        currentSensorCols = validSensorsToKeep

    # Step 5: Clip outliers using IQR method (1.5 * IQR rule)
    print("Clipping outliers using IQR...")
    q1 = DataFrameClean[currentSensorCols].quantile(0.25)
    q3 = DataFrameClean[currentSensorCols].quantile(0.75)
    iqr = q3 - q1
    for col in currentSensorCols:
        lb, ub = q1[col] - 1.5*iqr[col], q3[col] + 1.5*iqr[col]
        DataFrameClean[col] = np.clip(DataFrameClean[col], lb, ub)

    # Step 6: Standardize sensor columns (mean=0, std=1) - RUL stays raw
    print("Standardizing sensor data...")
    scaler = StandardScaler()
    DataFrameClean[currentSensorCols] = scaler.fit_transform(DataFrameClean[currentSensorCols])

    print(f"Final cleaned shape: {DataFrameClean.shape}")
    return DataFrameClean


def main(trainPath="train.xlsx", testPath="test.xlsx", rulPath="RUL.xlsx"):

    # Step 1: Load Excel datasets with headers
    print("Step 1: Loading Excel datasets WITH HEADERS...")
    train, test, rul = LoadDataset(trainPath, testPath, rulPath)

    # Step 2: Initial data quality assessment
    CheckDataIntegrity(train, "Raw Train Data")
    CheckDataIntegrity(test, "Raw Test Data")

    # Step 3: Calculate Remaining Useful Life (RUL) for training engines
    print("\nStep 2: Calculating RUL for Training Data...")
    maxCycles = train.groupby('ID')['Cycle'].max().reset_index()
    maxCycles.columns = ['ID', 'MaxCycle']
    train = train.merge(maxCycles, on='ID')
    train['RUL'] = train['MaxCycle'] - train['Cycle']
    train.drop('MaxCycle', axis=1, inplace=True)

    # Step 4: Clean training data with automatic sensor selection
    print("\nStep 3: Cleaning Train Data (Auto sensor selection)...")
    cleanedTrain = CleanDataBasic(train.copy(), sensorsToKeep=None)
    finalSensorCols = [col for col in cleanedTrain.columns if col.startswith('S')]

    # Step 5: Clean test data using same sensors (model consistency)
    print("\nStep 4: Cleaning Test Data (Using train sensors)...")
    cleanedTest = CleanDataBasic(test.copy(), sensorsToKeep=finalSensorCols)

    # Step 6: Merge true RUL values with test data (NASA standard)
    print("\nStep 5: Merging RUL with Test Data...")
    print(f"Test engines: {cleanedTest['ID'].nunique()}, RUL file rows: {len(rul)}")

    # Handle RUL file (single column)
    if len(rul.columns) == 1:
        rul.columns = ['RUL']
    else:
        rul = rul.iloc[:, 0]  # Take first column as RUL
        rul.name = 'RUL'
        rul = rul.to_frame()

    rul['ID'] = range(1, len(rul)+1)
    maxCyclesTest = cleanedTest.groupby('ID')['Cycle'].max().reset_index()
    maxCyclesTest.columns = ['ID', 'MaxCycle']

    cleanedTest = cleanedTest.merge(maxCyclesTest, on='ID', how='left')
    cleanedTest = cleanedTest.merge(rul, on='ID', how='left')
    cleanedTest['RUL'] = cleanedTest['RUL'].fillna(0) + (cleanedTest['MaxCycle'] - cleanedTest['Cycle'])
    cleanedTest.drop('MaxCycle', axis=1, inplace=True)

    # Step 7: Apply NASA RUL capping [0, 130] for consistent scoring
    print("\nStep 6: Applying NASA RUL Capping (max=130)...")
    max_rul = 130
    cleanedTrain['RUL'] = np.clip(cleanedTrain['RUL'], 0, max_rul)
    cleanedTest['RUL'] = np.clip(cleanedTest['RUL'], 0, max_rul)

    # Step 8: Final cleaned data validation
    print("\n" + "="*80)
    print("FINAL CLEANED DATASETS:")
    print(f"  Train shape: {cleanedTrain.shape}")
    print(f"  Test shape:  {cleanedTest.shape}")
    print(f"  Sensors kept: {finalSensorCols}")

    CheckDataIntegrity(cleanedTrain, "Final Cleaned Train")
    CheckDataIntegrity(cleanedTest, "Final Cleaned Test")


    # EDA & Visualization
    print("\n" + "="*70)
    print("CREATING 8 PROFESSIONAL EDA CHARTS")
    print("="*70)

    # Chart 1: Engine Lifespan
    print("\nChart 1: Engines Lifespan...")
    plt.figure(figsize=(12, 6))
    engine_lifespans = train.groupby('ID')['Cycle'].max()
    plt.hist(engine_lifespans, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(engine_lifespans.median(), color='red', linestyle='--', label=f'Median: {engine_lifespans.median():.0f}')
    plt.title('Engine Lifespan Distribution', fontweight='bold')
    plt.xlabel('Total Cycles'); plt.ylabel('Number of Engines')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig('chart1_lifespan.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Chart 2: Sensor Variance
    print("\nChart 2: Sensors Variance...")
    plt.figure(figsize=(12, 6))
    variances = cleanedTrain[finalSensorCols].var().sort_values(ascending=False)
    variances.plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Sensor Variance Ranking', fontweight='bold')
    plt.ylabel('Variance'); plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('chart2_variance.png', dpi=300, bbox_inches='tight')
    plt.show()


    # Chart 3: RUL Before/After
    print("\nChart 3: RUL Before/After Capping...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    train['RUL'].hist(bins=50, ax=ax1, alpha=0.7, color='red')
    ax1.axvline(130, color='orange', linestyle='--'); ax1.set_title('Before Capping')
    cleanedTrain['RUL'].hist(bins=30, ax=ax2, alpha=0.7, color='green')
    ax2.axvline(130, color='orange', linestyle='--'); ax2.set_title('After Capping [0,130]')
    plt.suptitle('RUL Distribution Comparison', fontweight='bold')
    plt.savefig('chart3_rul.png', dpi=300, bbox_inches='tight')
    plt.show()


    # Chart 4: Engine #1 Degradation Pattern (CLEANED DATA)
    print("\nChart 4: Engine #1 Degradation...")

    engine1 = cleanedTrain[cleanedTrain['ID'] == 1].sort_values('Cycle')
    top_sensors = finalSensorCols[:5]
    max_cycle = engine1['Cycle'].max()

    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_sensors)))

    sensor_lines = []
    for i, sensor in enumerate(top_sensors):
        sensor_data = engine1[sensor].fillna(method='ffill').fillna(0)
        line, = plt.plot(engine1['Cycle'], sensor_data, linewidth=2.5,
                         label=f'{sensor}', color=colors[i],
                         marker='o', markersize=2, alpha=0.9)
        sensor_lines.append(line)

    plt.axvline(max_cycle, color='darkred', linestyle=':', linewidth=4, alpha=0.9)

    handles = sensor_lines + [
        plt.Line2D([0], [0], color='darkred', linestyle=':', linewidth=4,
                   label=f'Engine Failure\n(Cycle {int(max_cycle)}, RUL=0)')
    ]

    plt.xlabel('Cycle Number', fontsize=12, fontweight='bold')
    plt.ylabel('Standardized Sensor Value (mean=0, std=1)', fontsize=12, fontweight='bold')
    plt.title('Engine #1: Multi-Sensor Degradation Patterns', fontsize=16, fontweight='bold', pad=20)

    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('chart4_engine1_degradation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


    # Chart 5: Sensor Correlation Heatmap (CLEANED  DATA)
    print("\nChart 5: Correlation Matrix...")
    plt.figure(figsize=(12, 10))
    corr_matrix = cleanedTrain[finalSensorCols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                linewidths=0.5)
    plt.title('Sensor Correlation Matrix (14 Selected Sensors)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('chart5_correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Count high correlations
    high_corr_pairs = np.where(np.abs(corr_matrix) > 0.9)

    print("\nHIGH CORRELATION CLEANUP...")
    high_corr = np.where(np.abs(corr_matrix) > 0.9)
    pairs = []
    for i, j in zip(high_corr[0], high_corr[1]):
        if i != j:
            sensor1, sensor2 = corr_matrix.index[i], corr_matrix.columns[j]
            corr_val = corr_matrix.loc[sensor1, sensor2]
            pairs.append((sensor1, sensor2, corr_val))

    print("High correlation pairs (>0.9):")
    for p in pairs:
        print(f"  {p[0]} - {p[1]}: {p[2]:.3f}")

    # DROP from DataFrames
    if pairs:
        to_remove = [pair[0] for pair in pairs[:1]]  # S9
        print(f"DROPPING COLUMNS FROM DATAFRAMES: {to_remove}")

        cleanedTrain = cleanedTrain.drop(columns=to_remove)
        cleanedTest  = cleanedTest.drop(columns=to_remove)

        # Update finalSensorCols
        finalSensorCols = [s for s in finalSensorCols if s not in to_remove]

        print(f"SHAPES AFTER DROP:")
        print(f"   Train: {cleanedTrain.shape}")
        print(f"   Test:  {cleanedTest.shape}")
        print(f"   Sensors: {len(finalSensorCols)}")
    else:
        print("   No high correlations!")

    print(f"   FINAL SENSORS: {finalSensorCols}")

    # SAVE CLEANED DATA AS EXCEL
    print("\n" + "="*80)
    print("SAVING CLEANED DATASETS AS EXCEL FILES")
    print("="*80)

    # Create Excel file
    with pd.ExcelWriter('cleaned_turbofan_data.xlsx', engine='openpyxl') as writer:
        cleanedTrain.to_excel(writer, sheet_name='Cleaned_Train', index=False)
        cleanedTest.to_excel(writer, sheet_name='Cleaned_Test', index=False)

    print("'cleaned_turbofan_data.xlsx' created!")

    # Download File in Device
    from google.colab import files
    files.download('cleaned_turbofan_data.xlsx')


    return cleanedTrain, cleanedTest, finalSensorCols

if __name__ == "__main__":

    # Excel file paths (with headers in First Row)
    trainPath = "train.xlsx"
    testPath = "test.xlsx"
    rulPath = "RUL.xlsx"

    # Run complete preprocessing pipeline
    cleanedTrain, cleanedTest, finalSensorCols = main(trainPath, testPath, rulPath)

    print("\nXGBOOST model training ready!")
    print(f"  XGBoost Train: {cleanedTrain.shape}")
    print(f"  XGBoost Test: {cleanedTest.shape}")
    print(f"  XGBoost features: {len(finalSensorCols)}")
    print(f"  Sensors: {finalSensorCols}")
    print(f"  Ready: X[{cleanedTrain.shape[0]}, {len(finalSensorCols)}]")