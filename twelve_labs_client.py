"""
Twelve Labs API client for multimodal video search.
"""

import time
from typing import Optional

import config


class TwelveLabsClient:
    """
    Wrapper around the Twelve Labs Video Understanding API.
    Provides index creation, video upload, and multimodal search.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.TWELVE_LABS_API_KEY
        self._client = None

        if self.api_key and self.api_key != "your_twelve_labs_api_key_here":
            try:
                from twelvelabs import TwelveLabs
                self._client = TwelveLabs(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "twelvelabs SDK not installed. Run: pip install twelvelabs"
                )
            except Exception as e:
                raise ConnectionError(f"Failed to initialise Twelve Labs client: {e}")

    @property
    def is_available(self) -> bool:
        """Check if the Twelve Labs client is configured and ready."""
        return self._client is not None

    def create_index(self, name: Optional[str] = None) -> str:
        """
        Create a new video index.

        Args:
            name: Index name. Defaults to config value.

        Returns:
            Index ID string.
        """
        if not self.is_available:
            raise RuntimeError("Twelve Labs client is not configured.")

        name = name or config.TWELVE_LABS_INDEX_NAME

        try:
            index = self._client.index.create(
                name=name,
                engines=[
                    {
                        "name": "marengo2.7",
                        "options": ["visual", "conversation", "text_in_video", "logo"],
                    }
                ],
            )
            return index.id
        except Exception as e:
            # If index already exists, try to find it
            if "already exists" in str(e).lower():
                return self._find_index(name)
            raise

    def _find_index(self, name: str) -> str:
        """Find an existing index by name."""
        page = 1
        while True:
            response = self._client.index.list(page=page)
            for idx in response.data:
                if idx.name == name:
                    return idx.id
            if not response.page_info.next_page:
                break
            page += 1
        raise RuntimeError(f"Index '{name}' not found.")

    def get_or_create_index(self, name: Optional[str] = None) -> str:
        """Get existing index or create a new one."""
        name = name or config.TWELVE_LABS_INDEX_NAME
        try:
            return self._find_index(name)
        except RuntimeError:
            return self.create_index(name)

    def upload_video(
        self,
        index_id: str,
        video_path: str,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> str:
        """
        Upload and index a video.

        Args:
            index_id: Target index ID.
            video_path: Path to the local video file.
            poll_interval: Seconds between status polls.
            timeout: Maximum wait time in seconds.

        Returns:
            Video/task ID string.
        """
        if not self.is_available:
            raise RuntimeError("Twelve Labs client is not configured.")

        task = self._client.task.create(index_id=index_id, file=video_path)
        task_id = task.id

        start_time = time.time()
        while True:
            task_status = self._client.task.retrieve(task_id)
            if task_status.status == "ready":
                return task_status.video_id
            if task_status.status == "failed":
                raise RuntimeError(
                    f"Video indexing failed: {getattr(task_status, 'error', 'unknown')}"
                )
            if time.time() - start_time > timeout:
                raise TimeoutError("Video indexing timed out.")
            time.sleep(poll_interval)

    def search(
        self,
        index_id: str,
        query: str,
        search_options: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search for moments in indexed videos.

        Args:
            index_id: Index to search.
            query: Natural language search query.
            search_options: Search modalities (e.g., ["visual", "conversation"]).
            top_k: Number of results.

        Returns:
            List of result dicts with start, end, score, video_id, etc.
        """
        if not self.is_available:
            raise RuntimeError("Twelve Labs client is not configured.")

        search_options = search_options or ["visual", "conversation", "text_in_video"]

        search_result = self._client.search.query(
            index_id=index_id,
            query_text=query,
            options=search_options,
        )

        results = []
        count = 0
        for clip in search_result.data:
            if count >= top_k:
                break
            results.append(
                {
                    "start": clip.start,
                    "end": clip.end,
                    "score": clip.score,
                    "video_id": clip.video_id,
                    "confidence": clip.confidence,
                    "metadata": getattr(clip, "metadata", {}),
                }
            )
            count += 1

        return results
