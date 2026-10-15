# Build a small static binary, run it on distroless as non-root.
FROM golang:1.24 AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /supportbot ./cmd/supportbot

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=build /supportbot /app/supportbot
COPY kb /app/kb
COPY data/bookings.json /app/data/bookings.json
ENV MODE=webhook PORT=8080 INDEX_PATH=/tmp/index.json
EXPOSE 8080
ENTRYPOINT ["/app/supportbot"]
