# =================================================================================================
# Name: Terraform Outputs
# Date: 2025-09-09
# Script Name: outputs.tf
# Version: 0.5.0
# Log Summary: Exposes primary outputs for user convenience.
# Description: References IP & DNS name from main.tf.
# Change Summary: Initial version.
# Inputs: Resources
# Outputs: IP & DNS
# =================================================================================================
output "instance_ip" {
  value = oci_core_instance.civic_api.public_ip
}

output "dns_hostname" {
  value = cloudflare_record.api_dns.hostname
}