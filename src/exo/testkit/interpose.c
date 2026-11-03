/*
 * Test kit only: send exo's discovery multicast to the harness's relay on the loopback.
 *
 * exo discovers peers with IPv6 multicast (rust/networking/src/discovery.rs). Nodes on one
 * machine can't use it: unicast replies to the shared discovery port reach only one of them,
 * and on macOS a process without Local Network access can't send multicast at all. Loaded into
 * a node (DYLD_INSERT_LIBRARIES on macOS, LD_PRELOAD on Linux), this turns every datagram sent
 * to exo's discovery group into a unicast datagram to [::1]:$EXO_TESTKIT_DISCOVERY_RELAY. The
 * relay hands each Hello to the other nodes and their replies back, so exo's discovery code
 * runs unchanged. Every other send is untouched.
 */
#ifndef __APPLE__
#define _GNU_SOURCE
#endif
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

static const unsigned char EXO_DISCOVERY_GROUP[16] = {
    0xff, 0x12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xe0, 0xa1, 0xde, 0x89};

static int relay_address(const struct sockaddr *to, socklen_t to_length,
                         struct sockaddr_in6 *relay) {
    if (to == NULL || to->sa_family != AF_INET6 ||
        to_length < (socklen_t)sizeof(struct sockaddr_in6)) {
        return 0;
    }
    const struct sockaddr_in6 *destination = (const struct sockaddr_in6 *)to;
    if (memcmp(&destination->sin6_addr, EXO_DISCOVERY_GROUP, 16) != 0) {
        return 0;
    }
    const char *port = getenv("EXO_TESTKIT_DISCOVERY_RELAY");
    if (port == NULL || *port == '\0') {
        return 0;
    }
    memset(relay, 0, sizeof *relay);
#ifdef __APPLE__
    relay->sin6_len = sizeof *relay;
#endif
    relay->sin6_family = AF_INET6;
    relay->sin6_addr = in6addr_loopback;
    relay->sin6_port = htons((unsigned short)atoi(port));
    return 1;
}

#ifdef __APPLE__

static ssize_t testkit_sendto(int fd, const void *buffer, size_t length, int flags,
                              const struct sockaddr *to, socklen_t to_length) {
    struct sockaddr_in6 relay;
    if (relay_address(to, to_length, &relay)) {
        return sendto(fd, buffer, length, flags, (const struct sockaddr *)&relay,
                      sizeof relay);
    }
    return sendto(fd, buffer, length, flags, to, to_length);
}

__attribute__((used)) static struct {
    const void *replacement;
    const void *replacee;
} interposers[] __attribute__((section("__DATA,__interpose"))) = {
    {(const void *)testkit_sendto, (const void *)sendto},
};

#else

#include <dlfcn.h>

typedef ssize_t (*sendto_function)(int, const void *, size_t, int, const struct sockaddr *,
                                   socklen_t);

ssize_t sendto(int fd, const void *buffer, size_t length, int flags,
               const struct sockaddr *to, socklen_t to_length) {
    static sendto_function real_sendto = NULL;
    if (real_sendto == NULL) {
        real_sendto = (sendto_function)dlsym(RTLD_NEXT, "sendto");
    }
    struct sockaddr_in6 relay;
    if (relay_address(to, to_length, &relay)) {
        return real_sendto(fd, buffer, length, flags, (const struct sockaddr *)&relay,
                           sizeof relay);
    }
    return real_sendto(fd, buffer, length, flags, to, to_length);
}

#endif
