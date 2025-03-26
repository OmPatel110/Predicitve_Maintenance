import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Previous loading and processing functions remain the same
@st.cache_resource
def load_models():
    """Load the machine failure prediction models"""
    try:
        with open("lr_machine_failure_model.pkl", "rb") as f:
            model1 = pickle.load(f)

        with open("rf_failure_type_models.pkl", "rb") as f:
            model2 = pickle.load(f)
            
        return model1, model2
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

def process_input_data(torque, rpm, process_temp, air_temp, type_val, tool_wear):
    """Process and engineer features from raw input"""
    try:
        # Calculate rotational speed in rad/s
        rot_speed_rads = rpm * (2 * np.pi / 60)
        
        # Calculate power
        power = torque * rot_speed_rads
        
        # Calculate temperature difference
        temp_diff = process_temp - air_temp
        
        # Create DataFrame with engineered features
        input_data = pd.DataFrame({
            'Type': [type_val],
            'Tool wear [min]': [tool_wear],
            'Power [W]': [power],
            'Temperature Difference [K]': [temp_diff]
        })
        
        # Convert numeric columns to float
        numeric_cols = ['Tool wear [min]', 'Power [W]', 'Temperature Difference [K]']
        input_data[numeric_cols] = input_data[numeric_cols].astype(float)
        
        return input_data
    
    except Exception as e:
        st.error(f"Error processing input data: {str(e)}")
        return None

def explain_prediction(model, input_data, class_names):
    """Generate LIME explanation for a single prediction"""
    try:
        # Generate synthetic training data
        n_samples = 1000
        
        # Get numeric columns
        numeric_cols = ['Tool wear [min]', 'Power [W]', 'Temperature Difference [K]']
        
        # Create ranges for numeric features
        ranges = {
            col: (input_data[col].iloc[0] * 0.5, input_data[col].iloc[0] * 1.5) 
            for col in numeric_cols
        }
        
        # Create synthetic data
        synthetic_data = {}
        
        # Generate numeric data
        for col in numeric_cols:
            min_val, max_val = ranges[col]
            synthetic_data[col] = np.random.uniform(min_val, max_val, n_samples)
            
        # Generate categorical data
        type_values = ['L', 'M', 'H']
        synthetic_data['Type'] = np.random.choice(type_values, n_samples)
        
        # Convert to DataFrame
        training_data = pd.DataFrame(synthetic_data)
        
        # Ensure same column order as input data
        training_data = training_data[input_data.columns]
        
        # Create categorical feature mapping
        type_mapping = {val: idx for idx, val in enumerate(type_values)}
        
        # Convert categorical values to numeric
        training_array = training_data.copy()
        training_array['Type'] = training_array['Type'].map(type_mapping)
        training_array = training_array.values
        
        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_array,
            feature_names=list(input_data.columns),
            categorical_features=[0],  # Index of Type column
            categorical_names={0: type_values},
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
        
        # Prediction wrapper function
        def predict_fn(instances):
            # Convert to DataFrame
            df = pd.DataFrame(instances, columns=input_data.columns)
            
            # Convert numeric Type values back to categorical
            df['Type'] = df['Type'].apply(lambda x: type_values[int(x)] if x < len(type_values) else type_values[0])
            
            # Ensure numeric columns are float
            for col in numeric_cols:
                df[col] = df[col].astype(float)
            
            # Get raw predictions
            raw_preds = model.predict_proba(df)
            
            # Handle 3D array output for multi-class prediction
            if isinstance(raw_preds, list) and len(raw_preds) == len(class_names):
                # This is likely one array per class in a list
                # Extract the positive class probability for each
                probs = np.column_stack([pred[:, 1] if pred.shape[1] > 1 else pred.flatten() 
                                        for pred in raw_preds])
                return probs
            
            # If it's already a 2D array, return it directly
            if isinstance(raw_preds, np.ndarray) and len(raw_preds.shape) == 2:
                return raw_preds
            
            # If it's a 3D array, reshape it to 2D
            if isinstance(raw_preds, np.ndarray) and len(raw_preds.shape) == 3:
                n_samples = raw_preds.shape[0]
                # Flatten the last two dimensions
                return raw_preds.reshape(n_samples, -1)
            
            # Last resort: try to convert to numpy array
            try:
                return np.array(raw_preds)
            except:
                # If all else fails, return dummy probabilities
                return np.zeros((len(df), len(class_names)))
                
        # Prepare input instance for explanation
        input_array = input_data.copy()
        input_array['Type'] = input_array['Type'].map(type_mapping)
        input_instance = input_array.values[0]  # Convert to numpy array
        
        # Get explanation
        exp = explainer.explain_instance(
            data_row=input_instance,
            predict_fn=predict_fn,
            num_features=len(input_data.columns),
            num_samples=100
        )
        
        return exp
        
    except Exception as e:
        st.error(f"Error in LIME explanation: {str(e)}")
        return None

def plot_lime_explanation(explanation, title):
    """Create and return a matplotlib figure with LIME explanation"""
    if explanation is None:
        return None
        
    try:
        # Create figure without passing ax
        fig = plt.figure(figsize=(10, 6))
        
        # Get the explanation as a list
        exp_list = explanation.as_list()
        
        # Sort features by absolute importance
        exp_list = sorted(exp_list, key=lambda x: abs(x[1]))
        
        # Separate features and values
        features = [x[0] for x in exp_list]
        values = [x[1] for x in exp_list]
        
        # Create colors based on positive/negative values
        colors = ['red' if x < 0 else 'blue' for x in values]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        plt.barh(y_pos, values, color=colors)
        
        # Customize the plot
        plt.yticks(y_pos, features)
        plt.xlabel('Impact on Prediction')
        plt.title(title)
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting LIME explanation: {str(e)}")
        return None

def analyze_input(input_data, model1, model2):
    """Analyze input data and return predictions with LIME explanations"""
    if model1 is None or model2 is None:
        st.error("Models not properly loaded.")
        return None
    
    if input_data is None:
        st.error("Invalid input data.")
        return None

    try:
        # Stage 1: Failure Detection
        stage1_pred_proba = model1.predict_proba(input_data)
        failure_prob = stage1_pred_proba[0][1]
        
        # Get Stage 1 LIME explanation
        stage1_exp = explain_prediction(
            model1, 
            input_data,
            class_names=['No Failure', 'Failure']
        )
        
        results = {
            'stage1_probability': failure_prob,
            'stage1_explanation': stage1_exp,
            'stage2_probability': None,
            'stage2_explanation': None
        }
        
        # If failure predicted, do stage 2 analysis
        if failure_prob > 0.5:
            # Get stage 2 prediction and adapt the result based on model output
            raw_stage2_pred = model2.predict_proba(input_data)
            
            # Handle different prediction formats
            if isinstance(raw_stage2_pred, list):
                # If it's a list of arrays (one per class), extract probabilities
                try:
                    stage2_probs = np.array([p[:, 1] for p in raw_stage2_pred]).flatten()
                except:
                    # If that fails, use the first array
                    stage2_probs = raw_stage2_pred[0]
            else:
                # If it's already a numpy array, use it directly
                stage2_probs = raw_stage2_pred[0]
            
            # Store the processed probabilities
            results['stage2_probability'] = stage2_probs
            
            # Get Stage 2 LIME explanation
            stage2_exp = explain_prediction(
                model2,
                input_data,
                class_names=['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            )
            results['stage2_explanation'] = stage2_exp
        
        return results
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.error(f"Detailed error: {str(e)}")
        return None

def display_failure_type_analysis(results):
    """Display the Stage 2 Failure Type Analysis with improved alignment"""
    # Define failure types and their full names
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    failure_full_names = {
        'TWF': 'Tool Wear Failure',
        'HDF': 'Heat Dissipation Failure',
        'PWF': 'Power Failure',
        'OSF': 'Overstrain Failure',
        'RNF': 'Random Failure'
    }
    
    probs = results['stage2_probability']
    
    # Ensure probs is the right length
    if len(probs) != len(failure_types):
        st.error(f"Expected {len(failure_types)} probabilities but got {len(probs)}!")
        # If possible, adapt the length
        if len(probs) > len(failure_types):
            probs = probs[:len(failure_types)]
        else:
            # Pad with zeros if too short
            probs = np.pad(probs, (0, len(failure_types) - len(probs)), 'constant')
    
    # Write header
    st.write("**Failure Type Probabilities:**")
    
    # Create a 5-column layout
    cols = st.columns(5)
    
    # Add content to each column with consistent formatting
    for i, (ftype, prob) in enumerate(zip(failure_types, probs)):
        try:
            # Convert to float and percentage
            prob_value = float(prob) * 100
            
            # Write to the appropriate column with consistent formatting
            with cols[i]:
                st.write(f"**{ftype}**")
                st.write(f"{failure_full_names[ftype]}")
                st.write(f"{prob_value:.1f}%")
                st.write(" ")  # Add some space
                
        except (ValueError, TypeError) as e:
            with cols[i]:
                st.write(f"**{ftype}**")
                st.write(f"{failure_full_names[ftype]}")
                st.write("N/A")
                st.write(" ")  # Add some space

def display_lime_results(results):
    """Display LIME results in both written and graphical format"""
    if not results:
        return
        
    # Stage 1 - Failure Detection
    st.write("### Stage 1 - Failure Detection:")
    
    # Get probabilities and explanations
    failure_prob = float(results['stage1_probability'])  # Convert to Python float
    stage1_exp = results['stage1_explanation']
    
    # Display prediction probabilities
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Prediction probabilities**")
        st.write(f"No Failure: {1-failure_prob:.2f}")
        st.write(f"Failure: {failure_prob:.2f}")
    
    # Format and display key factors
    if stage1_exp:
        feature_importance = stage1_exp.as_list()
        
        with col2:
            st.write(" ")
        
        # Display key factors in text format
        st.write("\n**Key Factors Contributing to Failure Prediction:**")
        for feature, value in feature_importance:
            sign = "increases" if value > 0 else "decreases"
            st.write(f"- {feature}: {sign} failure probability by {abs(value):.4f}")
        
        # Display LIME visualization
        st.write("\n**Local explanation for class Failure**")
        fig1 = plot_lime_explanation(stage1_exp, "Feature Importance")
        if fig1:
            st.pyplot(fig1)
    
    # Stage 2 - Failure Type Analysis (if failure predicted)
    if failure_prob > 0.5 and results['stage2_probability'] is not None:
        st.write("\n### Stage 2 - Failure Type Analysis:")
        
        # Use the improved display function for failure type analysis
        display_failure_type_analysis(results)
        
        # Define failure types for later use
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        failure_full_names = {
            'TWF': 'Tool Wear Failure',
            'HDF': 'Heat Dissipation Failure',
            'PWF': 'Power Failure',
            'OSF': 'Overstrain Failure',
            'RNF': 'Random Failure'
        }
        probs = results['stage2_probability']
        
        # Add recommendations for failure types with probability > 20%
        st.write("\n**Specific Recommendations:**")
        
        for ftype, prob in zip(failure_types, probs):
            prob_value = float(prob) * 100
            
            if prob_value > 50:
                if ftype == 'TWF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Tool Wear Failure:")
                    st.write("- Inspect tool for excessive wear and damage")
                    st.write("- Check if tool replacement is needed")
                    st.write("- Verify proper tool installation and alignment")
                    st.write("- Review cutting parameters for optimization")
                    
                elif ftype == 'HDF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Heat Dissipation Failure:")
                    st.write("- Check cooling systems and heat sinks")
                    st.write("- Inspect ventilation and airflow paths")
                    st.write("- Clean any dust or debris blocking cooling components")
                    st.write("- Verify proper operation of fans and cooling pumps")
                    
                elif ftype == 'PWF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Power Failure:")
                    st.write("- Inspect electrical connections and wiring")
                    st.write("- Check power supply units and regulators")
                    st.write("- Test for voltage fluctuations or instability")
                    st.write("- Verify proper grounding and surge protection")
                    
                elif ftype == 'OSF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Overstrain Failure:")
                    st.write("- Check for misalignment in moving parts")
                    st.write("- Inspect bearings and support structures")
                    st.write("- Verify proper lubrication of mechanical components")
                    st.write("- Look for signs of excessive vibration or imbalance")
                    
                elif ftype == 'RNF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Random Failure:")
                    st.write("- Perform comprehensive system diagnostics")
                    st.write("- Check for intermittent issues in all subsystems")
                    st.write("- Review maintenance history for patterns")
                    st.write("- Consider environmental factors that might contribute to random failures")
            
            elif prob_value > 20:
                if ftype == 'TWF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Tool Wear Failure (Moderate risk):")
                    st.write("- Monitor tool condition more frequently")
                    st.write("- Review tool wear patterns for abnormalities")
                    
                elif ftype == 'HDF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Heat Dissipation Failure (Moderate risk):")
                    st.write("- Monitor operating temperatures more closely")
                    st.write("- Schedule inspection of cooling systems")
                    
                elif ftype == 'PWF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Power Failure (Moderate risk):")
                    st.write("- Monitor power consumption patterns")
                    st.write("- Check for minor fluctuations in power supply")
                    
                elif ftype == 'OSF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Overstrain Failure (Moderate risk):")
                    st.write("- Monitor for unusual vibrations or noises")
                    st.write("- Check alignment during next scheduled maintenance")
                    
                elif ftype == 'RNF':
                    st.write(f"**{ftype} ({prob_value:.1f}%)** - Random Failure (Moderate risk):")
                    st.write("- Review environmental conditions")
                    st.write("- Check for unusual patterns in operation logs")

        # Display Stage 2 LIME explanation
        stage2_exp = results['stage2_explanation']
        if stage2_exp:
            st.write("\n**Key Factors Contributing to Failure Type:**")
            feature_importance = stage2_exp.as_list()
            for feature, value in feature_importance:
                st.write(f"- {feature}: impact = {value:.4f}")
            
            # Display LIME visualization for failure type
            fig2 = plot_lime_explanation(stage2_exp, "Feature Importance for Failure Type")
            if fig2:
                st.pyplot(fig2)
        
        # Determine most likely failure type and provide specific recommendations
        if len(probs) > 0:
            most_likely_idx = np.argmax(probs)
            most_likely_type = failure_types[most_likely_idx]
            most_likely_prob = float(probs[most_likely_idx]) * 100
            
            # Overall recommendation based on stage 1 probability
            st.write("\n**Overall Recommendation:**")
            if failure_prob > 0.8:
                st.error("⚠️ URGENT: Immediate maintenance required!")
            elif failure_prob > 0.5:
                st.warning("⚠️ Schedule maintenance soon")
            else:
                st.success("✅ Continue regular maintenance schedule")
            
def main():
    st.title("Machine Failure Analysis System with LIME Interpretation")
    
    # Load models
    model1, model2 = load_models()
    
    # Input form
    st.subheader("Enter Machine Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        machine_type = st.selectbox("Machine Type", options=['L', 'M', 'H'])
        torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0, step=10.0)
        rpm = st.number_input("Rotational Speed (rpm)", min_value=0.0, max_value=5000.0, value=1500.0, step=100.0)
    
    with col2:
        process_temp = st.number_input("Process Temperature (K)", min_value=180.0, max_value=500.0, value=310.0, step=10.0)
        air_temp = st.number_input("Air Temperature (K)", min_value=180.0, max_value=500.0, value=300.0, step=10.0)
        tool_wear = st.number_input("Tool Wear (minutes)", min_value=0.0, max_value=250.0, value=100.0, step=10.0)
    
    if st.button("Analyze"):
        # Process input data
        input_data = process_input_data(
            torque, rpm, process_temp, air_temp, machine_type, tool_wear
        )
        
        if input_data is not None:
            # Run analysis
            with st.spinner("Analyzing..."):
                results = analyze_input(input_data, model1, model2)
            
            if results:
                st.subheader("Analysis Results")
                # Call the function to display results
                display_lime_results(results)

if __name__ == "__main__":
    main()  
