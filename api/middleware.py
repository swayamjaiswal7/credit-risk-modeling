"""
Custom middleware for FastAPI application
Handles logging, request validation, and monitoring
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Callable
import json

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests and responses
    Tracks request duration and status codes
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details"""
        
        # Start timer
        start_time = time.time()
        
        # Log incoming request
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"Status: {response.status_code} Duration: {duration:.3f}s"
            )
            
            # Add custom headers
            response.headers["X-Process-Time"] = str(duration)
            response.headers["X-API-Version"] = "1.0.0"
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"Error: {str(e)} Duration: {duration:.3f}s",
                exc_info=True
            )
            raise


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request validation and sanitization
    Checks for common issues and malformed requests
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate request before processing"""
        
        # Check content type for POST requests
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            
            if not content_type.startswith("application/json"):
                from fastapi.responses import JSONResponse
                logger.warning(
                    f"Invalid content type: {content_type} for {request.url.path}"
                )
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "Unsupported Media Type",
                        "detail": "Content-Type must be application/json",
                        "received": content_type
                    }
                )
        
        # Process request
        response = await call_next(request)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware
    Limits requests per IP address
    
    Note: For production, use Redis-based rate limiting
    """
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        """
        Initialize rate limiter
        
        Args:
            app: FastAPI application
            calls: Number of allowed calls
            period: Time period in seconds
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = {}  # Store: {ip: [(timestamp, count), ...]}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit and process request"""
        
        # Get client IP
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_old_entries(current_time)
        
        # Check rate limit
        if client_ip in self.requests:
            request_count = sum(1 for ts in self.requests[client_ip] 
                              if current_time - ts < self.period)
            
            if request_count >= self.calls:
                from fastapi.responses import JSONResponse
                logger.warning(f"Rate limit exceeded for {client_ip}")
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate Limit Exceeded",
                        "detail": f"Maximum {self.calls} requests per {self.period} seconds",
                        "retry_after": self.period
                    },
                    headers={"Retry-After": str(self.period)}
                )
        
        # Add request timestamp
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.calls - len(self.requests.get(client_ip, []))
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.period))
        
        return response
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than the period"""
        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                ts for ts in self.requests[ip] 
                if current_time - ts < self.period
            ]
            if not self.requests[ip]:
                del self.requests[ip]


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware with additional security
    """
    
    def __init__(self, app, allowed_origins: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS preflight and add CORS headers"""
        
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Max-Age"] = "3600"
            return response
        
        # Process normal request
        response = await call_next(request)
        
        # Add CORS headers
        if origin and (self.allowed_origins == ["*"] or origin in self.allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect metrics
    Tracks request counts, latencies, and errors
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_duration = 0.0
        self.endpoint_stats = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each request"""
        
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            response = await call_next(request)
            
            # Update metrics
            duration = time.time() - start_time
            self.request_count += 1
            self.total_duration += duration
            
            # Track per-endpoint stats
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'errors': 0
                }
            
            self.endpoint_stats[endpoint]['count'] += 1
            self.endpoint_stats[endpoint]['total_duration'] += duration
            
            # Track errors
            if response.status_code >= 400:
                self.error_count += 1
                self.endpoint_stats[endpoint]['errors'] += 1
            
            return response
            
        except Exception as e:
            self.error_count += 1
            if endpoint in self.endpoint_stats:
                self.endpoint_stats[endpoint]['errors'] += 1
            raise
    
    def get_metrics(self) -> dict:
        """Get collected metrics"""
        avg_duration = (self.total_duration / self.request_count 
                       if self.request_count > 0 else 0)
        
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': (self.error_count / self.request_count 
                          if self.request_count > 0 else 0),
            'average_duration': avg_duration,
            'endpoint_stats': {
                endpoint: {
                    'count': stats['count'],
                    'avg_duration': (stats['total_duration'] / stats['count'] 
                                    if stats['count'] > 0 else 0),
                    'errors': stats['errors']
                }
                for endpoint, stats in self.endpoint_stats.items()
            }
        }


# Metrics instance (can be accessed for monitoring endpoints)
metrics_middleware = None


def get_metrics() -> dict:
    """Get current metrics"""
    if metrics_middleware:
        return metrics_middleware.get_metrics()
    return {
        'total_requests': 0,
        'total_errors': 0,
        'error_rate': 0,
        'average_duration': 0,
        'endpoint_stats': {}
    }