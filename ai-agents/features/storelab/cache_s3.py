"""S3 backend for the response cache (SL-08): entries under ``storelab/cache/``.

Outside core/ because it talks to S3. The client is any object with boto3's
``get_object``, ``put_object``, ``delete_object`` and ``list_objects_v2``; by default
``boto3.client("s3")`` (boto3 is not a StoreLab dependency yet, so it is imported
only when no client is given).
"""
from __future__ import annotations

from typing import Any, Iterator

PREFIX = "storelab/cache/"


class S3CacheStore:
    def __init__(self, bucket: str, prefix: str = PREFIX, client: Any = None):
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.bucket, self.prefix, self.client = bucket, prefix if prefix.endswith("/") else prefix + "/", client

    def get(self, path: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self.prefix + path)["Body"].read()
        except Exception as error:  # botocore ClientError; its code says whether the key is missing
            code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
            if code in ("NoSuchKey", "404"):
                return None
            raise

    def put(self, path: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.prefix + path, Body=data,
                               ContentType="application/json")

    def delete(self, path: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self.prefix + path)

    def paths(self) -> Iterator[str]:
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": self.prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            for item in page.get("Contents", []):
                path = item["Key"][len(self.prefix):]
                if path.endswith(".json") and path.count("/") >= 2:
                    yield path
            if not page.get("IsTruncated"):
                return
            token = page.get("NextContinuationToken")
