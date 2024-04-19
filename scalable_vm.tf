# "path/to/your/credentials.json", "your-project-id" need to be inserted before running
# region, zone, and instance count - modify as per need

provider "google" {
  credentials = file("path/to/your/credentials.json")
  project     = "your-project-id"
  region      = "us-west1"
}

resource "google_compute_instance_template" "caption_train_cluster"{
  name        = "caption_train_cluster"
  machine_type = "n1-standard-1"
  
  disk {
    source_image = "debian-cloud/debian-10"
    auto_delete  = true
  }
  
  network_interface {
    network = "default"
    access_config {}
  }

}

resource "google_compute_instance_group_manager" "example_instance_group_manager" {
  name        = "example-instance-group-manager"
  base_instance_name = "example-instance"
  instance_template = google_compute_instance_template.example_template.self_link
  zone = "us-central1-a"
  
  target_size = 2
}

resource "google_compute_http_health_check" "example_http_health_check" {
  name               = "example-http-health-check"
  request_path       = "/"
  check_interval_sec = 1
  timeout_sec        = 1
}

resource "google_compute_backend_service" "example_backend_service" {
  name = "example-backend-service"
  port_name = "http"
  protocol = "HTTP"
  
  backend {
    group = google_compute_instance_group_manager.example_instance_group_manager.self_link
  }
  
  health_checks = [google_compute_http_health_check.example_http_health_check.self_link]
}

resource "google_compute_url_map" "example_url_map" {
  name            = "example-url-map"
  default_service = google_compute_backend_service.example_backend_service.self_link
}

resource "google_compute_global_forwarding_rule" "example_forwarding_rule" {
  name       = "example-forwarding-rule"
  target     = google_compute_url_map.example_url_map.self_link
  port_range = "80"
}
