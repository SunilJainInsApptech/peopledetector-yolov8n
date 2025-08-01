# 🔒 Secure Twilio Environment Variable Configuration for Viam Agent

This guide explains how to securely configure Twilio credentials for fall detection alerts using Viam's built-in environment variable mechanism.

## Why Use Environment Variables?

✅ **Security**: Credentials are not stored in configuration files  
✅ **Version Control Safe**: No accidental credential commits  
✅ **Viam Native**: Uses Viam's official environment variable system  
✅ **Process-Level Security**: Variables are available only to viam-server and modules  

## Configuration Steps

### Step 1: Update Your Viam Robot Configuration

Edit your robot's JSON configuration file (e.g., `viam_config_fall_alerts.json`) to include the `agent.advanced_settings.viam_server_env` section:

```json
{
  "agent": {
    "advanced_settings": {
      "viam_server_env": {
        "TWILIO_ACCOUNT_SID": "your_actual_account_sid_here",
        "TWILIO_AUTH_TOKEN": "your_actual_auth_token_here",
        "TWILIO_FROM_PHONE": "+1234567890",
        "TWILIO_TO_PHONES": "+1987654321,+1555123456",
        "TWILIO_WEBHOOK_URL": "https://your-server.com/fall-images"
      }
    }
  },
  "services": [
    {
      "name": "yolov8-pose-detector",
      "type": "vision",
      "namespace": "rdk",
      "model": "rig-guardian:yolov8n-pose:yolov8n-pose",
      "attributes": {
        "model_location": "/path/to/yolov8n-pose.pt",
        "pose_classifier_path": "/path/to/pose_classifier.joblib",
        "use_env_for_twilio": true,
        "fall_confidence_threshold": 0.7,
        "alert_cooldown_seconds": 300
      }
    }
  ]
}
```

### Step 2: Replace Placeholder Values

Replace these placeholder values with your actual Twilio credentials:

- `TWILIO_ACCOUNT_SID`: Your Twilio Account SID (found in Twilio Console)
- `TWILIO_AUTH_TOKEN`: Your Twilio Auth Token (found in Twilio Console)  
- `TWILIO_FROM_PHONE`: Your Twilio phone number (e.g., "+15551234567")
- `TWILIO_TO_PHONES`: Comma-separated list of recipient phone numbers
- `TWILIO_WEBHOOK_URL`: (Optional) URL for receiving fall detection images

### Step 3: Apply Configuration

1. Save your configuration file
2. The viam-agent will automatically restart viam-server to apply the changes
3. Environment variables will be available to your YOLOv8 pose detection module

## Environment Variable Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `TWILIO_ACCOUNT_SID` | Yes | Twilio Account SID | `YOUR_ACCOUNT_SID_HERE` |
| `TWILIO_AUTH_TOKEN` | Yes | Twilio Auth Token | `your_auth_token` |
| `TWILIO_FROM_PHONE` | Yes | Twilio phone number | `+15551234567` |
| `TWILIO_TO_PHONES` | Yes | Recipient phone numbers (comma-separated) | `+19876543210,+15551234567` |
| `TWILIO_WEBHOOK_URL` | No | Webhook URL for image uploads | `https://example.com/webhooks/fall` |

## Configuration Attributes Reference

| Attribute | Default | Description |
|-----------|---------|-------------|
| `use_env_for_twilio` | `true` if `TWILIO_ACCOUNT_SID` env var exists | Enable environment variable loading |
| `fall_confidence_threshold` | `0.7` | Minimum confidence for fall detection trigger |
| `alert_cooldown_seconds` | `300` | Seconds between alerts for same person |

## How It Works

1. **viam-agent** reads the `viam_server_env` configuration
2. **viam-server** is launched with these environment variables  
3. **YOLOv8 module** accesses variables via `os.environ.get()`
4. **Fall detection** triggers SMS alerts using secure credentials

## Troubleshooting

### Check Environment Variable Status

The module logs will show the status of environment variables:

```
🔒 LOADING TWILIO CREDENTIALS FROM ENVIRONMENT VARIABLES (secure method)
Environment variable status: {'TWILIO_ACCOUNT_SID': 'SET', 'TWILIO_AUTH_TOKEN': 'SET', ...}
```

### Common Issues

**Environment variables show as 'NOT_SET':**
- Verify the `agent.advanced_settings.viam_server_env` section is properly formatted
- Check that viam-agent restarted after configuration changes
- Ensure credentials are not empty strings

**Fall alerts not working:**
- Verify all required environment variables are 'SET'
- Check Twilio credentials are valid in Twilio Console
- Verify phone numbers are in E.164 format (+1234567890)

**Module not loading environment variables:**
- Ensure `use_env_for_twilio: true` is set in service attributes
- Check that the service is using the correct module version

## Security Best Practices

✅ **Never commit actual credentials** to version control  
✅ **Use the template file** (`viam_config_fall_alerts_template.json`) for sharing  
✅ **Rotate credentials regularly** through Twilio Console  
✅ **Monitor Twilio usage** for unexpected activity  
✅ **Use webhook URLs with HTTPS** only  

## Migration from Config-Based Credentials

If you previously had credentials in the service attributes:

1. Copy your actual values from the old configuration
2. Add them to `agent.advanced_settings.viam_server_env`  
3. Remove the old `twilio_*` attributes from service configuration
4. Set `use_env_for_twilio: true`
5. Save and let viam-agent restart

The module supports both methods for backwards compatibility, but environment variables take precedence when available.

## Template Files

- `viam_config_fall_alerts_template.json`: Template with placeholders
- `viam_config_fall_alerts.json`: Working example (replace values)

---

## Example Complete Configuration

```json
{
  "agent": {
    "advanced_settings": {
      "viam_server_env": {
        "TWILIO_ACCOUNT_SID": "YOUR_TWILIO_ACCOUNT_SID_HERE",
        "TWILIO_AUTH_TOKEN": "your_secret_auth_token_here",
        "TWILIO_FROM_PHONE": "+15551234567",
        "TWILIO_TO_PHONES": "+19876543210,+15559876543",
        "TWILIO_WEBHOOK_URL": "https://your-domain.com/fall-alerts"
      }
    }
  },
  "modules": [
    {
      "name": "yolov8n-pose",
      "type": "registry", 
      "module_id": "rig-guardian:yolov8n-pose:yolov8n-pose",
      "version": "latest"
    }
  ],
  "services": [
    {
      "name": "yolov8-pose-detector",
      "type": "vision",
      "namespace": "rdk",
      "model": "rig-guardian:yolov8n-pose:yolov8n-pose",
      "attributes": {
        "model_location": "/home/user/yolov8n-pose.pt",
        "pose_classifier_path": "/home/user/pose_classifier.joblib", 
        "use_env_for_twilio": true,
        "fall_confidence_threshold": 0.7,
        "alert_cooldown_seconds": 300
      }
    }
  ],
  "components": [
    {
      "name": "camera",
      "type": "camera",
      "namespace": "rdk", 
      "model": "webcam",
      "attributes": {
        "video_path": "0"
      }
    }
  ]
}
```

This configuration provides secure, production-ready fall detection alerts with proper credential management! 🚨📱
