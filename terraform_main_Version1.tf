# =================================================================================================
# Name: Terraform Main
# Date: 2025-09-09
# Script Name: main.tf
# Version: 0.5.0
# Log Summary: OCI compute + Cloudflare DNS example scaffold.
# Description: Provision an OCI instance and Cloudflare DNS record pointing at it.
# Change Summary: Initial version (minimal).
# Inputs: Variables in variables.tf
# Outputs: Public IP, DNS record
# =================================================================================================
terraform {
  required_providers {
    oci = {
      source = "oracle/oci"
    }
    cloudflare = {
      source = "cloudflare/cloudflare"
    }
  }
  required_version = ">= 1.5.0"
}

provider "oci" {
  tenancy_ocid        = var.oci_tenancy_ocid
  user_ocid           = var.oci_user_ocid
  fingerprint         = var.oci_fingerprint
  private_key_path    = var.oci_private_key_path
  region              = var.oci_region
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

resource "oci_core_instance" "civic_api" {
  availability_domain = var.oci_ad
  compartment_id      = var.oci_compartment_ocid
  shape               = var.oci_shape
  display_name        = "civic-legis-api"
  source_details {
    source_type = "image"
    source_id   = var.oci_image_ocid
  }
  create_vnic_details {
    subnet_id = var.oci_subnet_ocid
    assign_public_ip = true
  }
  metadata = {
    ssh_authorized_keys = file(var.ssh_public_key)
  }
}

resource "cloudflare_record" "api_dns" {
  zone_id = var.cloudflare_zone_id
  name    = var.api_subdomain
  value   = oci_core_instance.civic_api.public_ip
  type    = "A"
  proxied = true
}

output "api_public_ip" {
  value = oci_core_instance.civic_api.public_ip
}
output "api_dns_record" {
  value = cloudflare_record.api_dns.hostname
}