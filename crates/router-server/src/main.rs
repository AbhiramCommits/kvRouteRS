#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    router_server::serve().await
}
