import os
from kfp import dsl
from kfp import compiler

# Define the data preparation component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=["google-cloud-bigquery", "db-dtypes"]
)
def prepare_training_data(
    project_id: str,
    static_training_data_gcs_uri: str,
    bigquery_table_uri: str,
    prepared_data_output_gcs_uri: str,
):
    from google.cloud import bigquery
    from google.cloud import storage
    import pandas as pd
    from urllib.parse import urlparse
    import os

    fallback = False
    if bigquery_table_uri and bigquery_table_uri.strip():
        print(f"Preparing data from BigQuery table: {bigquery_table_uri} in project {project_id}")
        try:
            client = bigquery.Client(project=project_id)
            query = f"""
                SELECT
                  CAST(SPLIT(TRIM(payload, '[]'), ',')[OFFSET(0)] AS FLOAT64) AS feature1,
                  CAST(SPLIT(TRIM(payload, '[]'), ',')[OFFSET(1)] AS FLOAT64) AS feature2
                FROM
                  `{bigquery_table_uri}`,
                  UNNEST(request_payload) AS payload
                WHERE
                  logging_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
            """
            print(f"Running query:\n{query}")
            df_bq = client.query(query).to_dataframe()
            print(f"Query returned {len(df_bq)} rows.")
            if len(df_bq) == 0:
                print("Warning: BigQuery returned 0 rows. Falling back to static training data.")
                fallback = True
            else:
                # Combine static baseline data and new BigQuery logging data
                print(f"Loading static baseline data from {static_training_data_gcs_uri}")
                df_static = pd.read_csv(static_training_data_gcs_uri)
                df = pd.concat([df_static, df_bq], ignore_index=True)
                print(f"Combined dataset has {len(df)} total rows.")
        except Exception as e:
            print(f"Error querying BigQuery: {e}. Falling back to static training data.")
            fallback = True
    else:
        print("No BigQuery table provided. Preparing data from static GCS URI.")
        fallback = True

    if fallback:
        print(f"Copying static training data from {static_training_data_gcs_uri} to {prepared_data_output_gcs_uri}")
        df = pd.read_csv(static_training_data_gcs_uri)
    
    parsed_url = urlparse(prepared_data_output_gcs_uri)
    bucket_name = parsed_url.netloc
    blob_path = parsed_url.path.lstrip("/")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    csv_data = df.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type="text/csv")
    print(f"Prepared training data saved to {prepared_data_output_gcs_uri}")


# Define the training component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=[]
)
def train_isolation_forest(
    training_data_gcs_uri: str,
    model_output_gcs_uri: str,
):
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    import joblib
    from google.cloud import storage
    from urllib.parse import urlparse
    import os

    print(f"Loading training data from {training_data_gcs_uri}")
    df = pd.read_csv(training_data_gcs_uri)
    
    # Train the Isolation Forest model
    # Assuming all columns except potentially a label or ID are features.
    # In unsupervised anomaly detection, we train only on the features.
    print("Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(df)
    
    # Save the model locally
    model_filename = "model.joblib"
    joblib.dump(model, model_filename)
    print(f"Saved model locally to {model_filename}")
    
    # Upload to Cloud Storage
    parsed_url = urlparse(model_output_gcs_uri)
    bucket_name = parsed_url.netloc
    blob_path = parsed_url.path.lstrip("/")
    
    # Ensure blob path ends with a slash to treat it as a directory, then append filename
    if blob_path and not blob_path.endswith("/"):
        blob_path += "/"
    dest_blob_name = os.path.join(blob_path, model_filename)
    
    print(f"Uploading model to gs://{bucket_name}/{dest_blob_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(model_filename)
    
    print("Model upload completed successfully.")


# Define the model upload and deploy component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=[]
)
def deploy_model_to_endpoint(
    project_id: str,
    region: str,
    model_display_name: str,
    endpoint_display_name: str,
    model_gcs_uri: str,
    serving_container_image_uri: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # Check if endpoint exists, otherwise create it
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc"
    )
    
    if endpoints:
        endpoint = endpoints[0]
        print(f"Found existing endpoint: {endpoint.resource_name}")
    else:
        print(f"Creating new endpoint: {endpoint_display_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=project_id,
            location=region
        )
        print(f"Created endpoint: {endpoint.resource_name}")

    # Register/Upload the model to Model Registry
    # Check if a model with the same display name exists to create a new version
    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc"
    )
    
    parent_model = models[0].resource_name if models else None
    
    print(f"Uploading model to registry: {model_display_name}")
    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_gcs_uri,
        serving_container_image_uri=serving_container_image_uri,
        parent_model=parent_model,
        is_default_version=True,
    )
    print(f"Uploaded model: {uploaded_model.resource_name}")

    # Deploy the model to the endpoint.
    # Note: For simplicity and continuous delivery, we route 100% traffic to the new model
    # and undeploy older models deployed to this endpoint.
    print(f"Deploying model {uploaded_model.resource_name} to endpoint {endpoint.resource_name}...")
    
    # List currently deployed models to undeploy them later
    deployed_models_to_remove = endpoint.list_models()
    
    # Deploy the new model
    endpoint.deploy(
        model=uploaded_model,
        deployed_model_display_name=model_display_name,
        traffic_percentage=100,
        machine_type="n1-standard-2", # Minimum machine type for Gemini Enterprise Agent Platform predictions
        min_replica_count=1,
        max_replica_count=1,
    )
    print("Model deployed successfully.")
    
    # Undeploy older models to clean up resources
    # We keep only the model we just deployed (which matches the new model resource name)
    for deployed_model in endpoint.list_models():
        if deployed_model.model != uploaded_model.resource_name:
            print(f"Undeploying old model deployment ID: {deployed_model.id} (Model: {deployed_model.model})")
            endpoint.undeploy(deployed_model_id=deployed_model.id)
            
    print("Cleanup of older deployed models completed.")
    
    # Find and return the newly deployed model ID
    new_deployed_model_id = None
    for deployed_model in endpoint.list_models():
        if deployed_model.model == uploaded_model.resource_name and deployed_model.model_version_id == uploaded_model.version_id:
            new_deployed_model_id = deployed_model.id
            break
            
    if not new_deployed_model_id:
        raise ValueError("Could not find the newly deployed model ID on the endpoint.")
        
    print(f"Newly deployed model ID: {new_deployed_model_id}")
    return new_deployed_model_id


# Define the model monitoring configuration component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=["google-cloud-aiplatform>=1.60.0"]
)
def configure_model_monitoring(
    project_id: str,
    region: str,
    model_display_name: str,
    endpoint_display_name: str,
    training_data_gcs_uri: str,
    skew_threshold: float,
):
    from google.cloud import aiplatform
    from google.cloud import aiplatform_v1beta1

    # Initialize client for regional endpoint
    client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    mm_client = aiplatform_v1beta1.ModelMonitoringServiceClient(client_options=client_options)
    schedule_client = aiplatform_v1beta1.ScheduleServiceClient(client_options=client_options)

    # Initialize standard SDK
    aiplatform.init(project=project_id, location=region)

    # 1. Resolve Model Registry name
    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc"
    )
    if not models:
        raise ValueError(f"Model '{model_display_name}' not found.")
    model_resource_name = models[0].resource_name

    # 2. Resolve Endpoint name
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc"
    )
    if not endpoints:
        raise ValueError(f"Endpoint '{endpoint_display_name}' not found.")
    endpoint = endpoints[0]
    endpoint_resource_name = endpoint.resource_name

    monitor_display_name = f"{model_display_name}-monitor"
    schedule_display_name = f"{model_display_name}-skew-schedule"
    parent_path = f"projects/{project_id}/locations/{region}"

    # 3. Delete existing Schedule if it exists to allow recreating with the new ModelMonitor
    list_schedules_req = aiplatform_v1beta1.ListSchedulesRequest(parent=parent_path)
    for schedule in schedule_client.list_schedules(request=list_schedules_req):
        if schedule.display_name == schedule_display_name:
            print(f"Deleting existing Schedule targeting older model: {schedule.name}")
            schedule_client.delete_schedule(name=schedule.name)

    # 4. Delete existing ModelMonitor if it exists to allow recreating targeting the new model version
    list_monitors_req = aiplatform_v1beta1.ListModelMonitorsRequest(parent=parent_path)
    for monitor in mm_client.list_model_monitors(request=list_monitors_req):
        if monitor.display_name == monitor_display_name:
            print(f"Deleting existing ModelMonitor targeting older model: {monitor.name}")
            operation = mm_client.delete_model_monitor(name=monitor.name)
            operation.result()

    # 5. Create new ModelMonitor targeting the newly trained model version
    print(f"Creating ModelMonitor: {monitor_display_name} targeting version {models[0].version_id}...")
    
    # Define Schema (feature order mapping for array predictions)
    mm_schema = aiplatform_v1beta1.ModelMonitoringSchema(
        feature_fields=[
            aiplatform_v1beta1.ModelMonitoringSchema.FieldSchema(name="feature1", data_type="float"),
            aiplatform_v1beta1.ModelMonitoringSchema.FieldSchema(name="feature2", data_type="float")
        ]
    )

    model_monitor = aiplatform_v1beta1.ModelMonitor(
        display_name=monitor_display_name,
        model_monitoring_target=aiplatform_v1beta1.ModelMonitor.ModelMonitoringTarget(
            vertex_model=aiplatform_v1beta1.ModelMonitor.ModelMonitoringTarget.VertexModelSource(
                model=model_resource_name,
                model_version_id=models[0].version_id
            )
        ),
        model_monitoring_schema=mm_schema,
        training_dataset=aiplatform_v1beta1.ModelMonitoringInput(
            columnized_dataset=aiplatform_v1beta1.ModelMonitoringInput.ModelMonitoringDataset(
                gcs_source=aiplatform_v1beta1.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource(
                    gcs_uri=training_data_gcs_uri,
                    format_=aiplatform_v1beta1.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat.CSV
                )
            )
        ),
        notification_spec=aiplatform_v1beta1.ModelMonitoringNotificationSpec(
            email_config=aiplatform_v1beta1.ModelMonitoringNotificationSpec.EmailConfig(
                user_emails=["darenwkt@google.com"]
            ),
            enable_cloud_logging=True
        )
    )

    create_monitor_req = aiplatform_v1beta1.CreateModelMonitorRequest(
        parent=parent_path,
        model_monitor=model_monitor
    )
    operation = mm_client.create_model_monitor(request=create_monitor_req)
    monitor_res = operation.result()
    existing_monitor_name = monitor_res.name
    print(f"Successfully created ModelMonitor: {existing_monitor_name}")

    # 6. Create Schedule since the older one was deleted
    if True:

        print(f"Creating Schedule for 15-minute intervals: {schedule_display_name}...")

        # Define Tabular Objective Spec
        objective_spec = aiplatform_v1beta1.ModelMonitoringObjectiveSpec(
            baseline_dataset=aiplatform_v1beta1.ModelMonitoringInput(
                columnized_dataset=aiplatform_v1beta1.ModelMonitoringInput.ModelMonitoringDataset(
                    gcs_source=aiplatform_v1beta1.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource(
                        gcs_uri=training_data_gcs_uri,
                        format_=aiplatform_v1beta1.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat.CSV
                    )
                )
            ),
            target_dataset=aiplatform_v1beta1.ModelMonitoringInput(
                vertex_endpoint_logs=aiplatform_v1beta1.ModelMonitoringInput.VertexEndpointLogs(
                    endpoints=[endpoint_resource_name]
                )
            ),
            tabular_objective=aiplatform_v1beta1.ModelMonitoringObjectiveSpec.TabularObjective(
                feature_drift_spec=aiplatform_v1beta1.ModelMonitoringObjectiveSpec.DataDriftSpec(
                    features=["feature1", "feature2"],
                    numeric_metric_type="jensen_shannon_divergence",
                    default_numeric_alert_condition=aiplatform_v1beta1.ModelMonitoringAlertCondition(
                        threshold=skew_threshold
                    )
                )
            )
        )

        job_request = aiplatform_v1beta1.CreateModelMonitoringJobRequest(
            parent=existing_monitor_name,
            model_monitoring_job=aiplatform_v1beta1.ModelMonitoringJob(
                display_name=f"{model_display_name}-monitoring-run",
                model_monitoring_spec=aiplatform_v1beta1.ModelMonitoringSpec(
                    objective_spec=objective_spec
                )
            )
        )

        # Define Cron Schedule (every 15 minutes)
        schedule_spec = aiplatform_v1beta1.Schedule(
            display_name=schedule_display_name,
            cron="*/15 * * * *",
            max_concurrent_run_count=1,
            create_model_monitoring_job_request=job_request
        )

        create_schedule_req = aiplatform_v1beta1.CreateScheduleRequest(
            parent=parent_path,
            schedule=schedule_spec
        )
        schedule_res = schedule_client.create_schedule(request=create_schedule_req)
        print(f"Successfully created Schedule: {schedule_res.name}")


# Define the KFP Pipeline
@dsl.pipeline(
    name="isolation-forest-anomaly-pipeline",
    description="Pipeline that trains an Isolation Forest anomaly detection model and deploys it to a Gemini Enterprise Agent Platform Endpoint"
)
def anomaly_detection_pipeline(
    project_id: str,
    region: str,
    training_data_gcs_uri: str,
    model_output_gcs_uri: str,
    model_display_name: str,
    endpoint_display_name: str,
    serving_container_image_uri: str,
    skew_threshold: float,
    predict_schema_gcs_uri: str,
    bigquery_table_uri: str = "",
):
    prepared_data_gcs_uri = f"{model_output_gcs_uri}_prepared_training_data.csv"

    # 1. Prepare the data (fetch from BigQuery if available, else static fallback)
    prepare_task = prepare_training_data(
        project_id=project_id,
        static_training_data_gcs_uri=training_data_gcs_uri,
        bigquery_table_uri=bigquery_table_uri,
        prepared_data_output_gcs_uri=prepared_data_gcs_uri,
    )

    # 2. Train the model
    train_task = train_isolation_forest(
        training_data_gcs_uri=prepared_data_gcs_uri,
        model_output_gcs_uri=model_output_gcs_uri,
    )
    train_task.after(prepare_task)
    
    # 3. Upload and Deploy the model
    deploy_task = deploy_model_to_endpoint(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        endpoint_display_name=endpoint_display_name,
        model_gcs_uri=model_output_gcs_uri,
        serving_container_image_uri=serving_container_image_uri,
    )
    
    deploy_task.after(train_task)

    # 3. Configure Model Monitoring
    monitoring_task = configure_model_monitoring(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        endpoint_display_name=endpoint_display_name,
        training_data_gcs_uri=training_data_gcs_uri,
        skew_threshold=skew_threshold,
    )

    monitoring_task.after(deploy_task)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="pipeline.yaml", help="Path to output the compiled pipeline spec")
    args = parser.parse_args()
    
    # Compile pipeline to a YAML file
    compiler.Compiler().compile(
        pipeline_func=anomaly_detection_pipeline,
        package_path=args.output
    )
    print(f"Pipeline compiled successfully to {args.output}")
