package io.tether.qvac.sdk

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.selects.select
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import to.holepunch.barerpc.BareRpc

/** The worker broke the JSON contract on top of bare-rpc. */
class QvacWorkerProtocolException(message: String) : Exception(message)

/**
 * JSON requests to the worker over one bare-rpc connection.
 *
 * [read] returns the next bytes the transport received, or null once it ends.
 * [send] writes one frame. The read loop runs until the transport ends, a
 * frame cannot be read, or [close] is called; the transport itself is closed
 * by its owner.
 */
class WorkerRpc(
    read: suspend () -> ByteArray?,
    send: suspend (ByteArray) -> Unit,
) {
    private val rpc = BareRpc(send = send)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val json = Json {
        ignoreUnknownKeys = true
        explicitNulls = false
    }

    init {
        scope.launch {
            try {
                while (true) rpc.receive(read() ?: break)
            } catch (_: Throwable) {
                // A frame receive already failed the connection with its cause; a
                // transport read error means the owner is closing it.
            } finally {
                withContext(NonCancellable) { rpc.close() }
            }
        }
    }

    suspend fun call(payload: JsonObject): JsonObject = decode(rpc.request(0u, encode(payload)))

    fun stream(payload: JsonObject): Flow<JsonObject> = decodeLines(rpc.requestFlow(0u, encode(payload)))

    /**
     * The payload is the first chunk on the request stream, then [input]. The
     * worker reads its duplex session's input until that stream ends, so a
     * cancelled collection ends it before the reply is abandoned. The request
     * runs in this connection's scope: cancelling the collector must not
     * cancel the request before its input has ended. Replies are handed over
     * one at a time, so the library's flow control sees how far the collector
     * has got; once the collector stops, they are dropped.
     */
    fun duplex(payload: JsonObject, input: Flow<ByteArray>): Flow<JsonObject> = decodeLines(flow {
        val stop = CompletableDeferred<Unit>()
        val body = flow {
            emit(encode(payload))
            emitAll(input.until(stop))
        }
        val replies = Channel<ByteArray>(Channel.RENDEZVOUS)
        val reply = scope.launch {
            try {
                rpc.requestFlow(0u, body).collect { chunk ->
                    select {
                        replies.onSend(chunk) {}
                        stop.onJoin {}
                    }
                }
                replies.close()
            } catch (error: Throwable) {
                replies.close(error)
            }
        }
        try {
            for (chunk in replies) emit(chunk)
        } finally {
            withContext(NonCancellable) {
                stop.complete(Unit)
                withTimeoutOrNull(REPLY_AFTER_INPUT_END_MS) { reply.join() }
                reply.cancel()
            }
        }
    })

    /** Fails pending and later calls; the read loop and in-flight requests end on their own. */
    suspend fun close() = rpc.close()

    private fun decodeLines(chunks: Flow<ByteArray>): Flow<JsonObject> = flow {
        val decoder = JsonLinesDecoder()
        chunks.collect { chunk ->
            for (line in decoder.feed(chunk)) emit(decode(line.encodeToByteArray()))
        }
        for (line in decoder.finish()) emit(decode(line.encodeToByteArray()))
    }

    private fun encode(payload: JsonObject): ByteArray {
        return json.encodeToString(JsonObject.serializer(), payload).encodeToByteArray()
    }

    private fun decode(payload: ByteArray): JsonObject {
        return try {
            json.parseToJsonElement(payload.decodeToString()).jsonObject
        } catch (error: Throwable) {
            throw QvacWorkerProtocolException("worker returned invalid JSON: ${error.message}")
        }
    }

    private fun Flow<ByteArray>.until(stop: Deferred<Unit>): Flow<ByteArray> = channelFlow {
        val producer = launch { collect { send(it) } }
        val stopper = launch {
            stop.await()
            producer.cancel()
        }
        producer.join()
        stopper.cancel()
    }

    private companion object {
        const val REPLY_AFTER_INPUT_END_MS = 5_000L
    }
}
