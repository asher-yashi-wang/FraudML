import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/**
 * JDK 11+ client for the local FraudML scoring proof of concept.
 * Reads a UTF-8 JSON request file so transaction data stays out of shell arguments.
 */
public final class FraudClient {
    private FraudClient() {}

    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length != 2) {
            System.err.println("Usage: java FraudClient http://127.0.0.1:8080 request.json");
            System.exit(2);
        }
        URI base = URI.create(args[0]);
        if (!"http".equals(base.getScheme()) && !"https".equals(base.getScheme())) {
            throw new IllegalArgumentException("Service URL must use HTTP or HTTPS");
        }
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(2))
                .build();
        // Each entry in transactions is one payment. GET /health lists the required features and types.
        HttpRequest request = HttpRequest.newBuilder(base.resolve("/score"))
                .timeout(Duration.ofSeconds(5))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(
                        Files.readString(Path.of(args[1]), StandardCharsets.UTF_8),
                        StandardCharsets.UTF_8))
                .build();
        HttpResponse<String> response = client.send(
                request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        // A timeout throws an exception. An HTTP error exits so the caller can handle the failure.
        if (response.statusCode() != 200) {
            System.err.println("Scoring failed with HTTP " + response.statusCode());
            System.exit(1);
        }
        // The response includes model_version and scores in request order for the caller's risk policy.
        System.out.println(response.body());
    }
}
