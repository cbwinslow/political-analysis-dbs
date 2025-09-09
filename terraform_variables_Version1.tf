# =================================================================================================
# Name: Terraform Variables
# Date: 2025-09-09
# Script Name: variables.tf
# Version: 0.5.0
# Log Summary: Input variable definitions.
# Description: Adjust according to your OCI & Cloudflare environment.
# Change Summary: Initial version.
# Inputs: Provided via tfvars or CLI.
# Outputs: Variables for main.tf
# =================================================================================================
variable "oci_tenancy_ocid" {}
variable "oci_user_ocid" {}
variable "oci_fingerprint" {}
variable "oci_private_key_path" {}
variable "oci_region" { default = "us-ashburn-1" }
variable "oci_compartment_ocid" {}
variable "oci_ad" { description = "Availability domain" }
variable "oci_shape" { default = "VM.Standard.A1.Flex" }
variable "oci_image_ocid" { description = "Oracle Linux or Ubuntu image OCID" }
variable "oci_subnet_ocid" {}
variable "ssh_public_key" {}
variable "cloudflare_api_token" {}
variable "cloudflare_zone_id" {}
variable "api_subdomain" { default = "civic-api" }